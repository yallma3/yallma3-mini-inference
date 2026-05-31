/*
 * yaLLMa3 - Framework for building AI agents that are capable of learning from their environment and interacting with it.
 * 
 * Copyright (C) 2025 yaLLMa3
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
 * 
 * This software is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied.
 * See the Mozilla Public License for the specific language governing permissions and limitations under the License.
 */


use crate::quantization::{quantize as q8_quantize, MutableQuantizedTensorQ8};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    Q8_0,
    Q4_0,
    F16,
}

#[derive(Debug)]
pub struct MutableQuantizedTensorQ4_0 {
    pub quant_vals: Vec<i8>,
    pub scale_factor: Vec<f32>,
}

#[derive(Debug)]
pub enum QuantizedTensor {
    Q8(MutableQuantizedTensorQ8),
    Q4_0(MutableQuantizedTensorQ4_0),
    F16(Vec<f32>),
}

impl QuantizedTensor {
    pub fn as_q8(&self) -> &MutableQuantizedTensorQ8 {
        match self {
            QuantizedTensor::Q8(ref q) => q,
            _ => panic!("as_q8 called on a non-Q8 tensor"),
        }
    }
    pub fn as_q8_mut(&mut self) -> &mut MutableQuantizedTensorQ8 {
        match self {
            QuantizedTensor::Q8(ref mut q) => q,
            _ => panic!("as_q8_mut called on a non-Q8 tensor"),
        }
    }
}

pub(crate) fn q4_0_quantize(qt: &mut MutableQuantizedTensorQ4_0, x: &[f32], n: usize, gs: u32) {
    let num_groups = n as u32 / gs;
    let q_max: f32 = 7.0f32;

    for group in 0..num_groups {
        let mut wmax: f32 = 0.0;
        for i in 0..gs {
            let val = x[(group * gs + i) as usize].abs();
            if val > wmax {
                wmax = val;
            }
        }

        let scale = wmax / q_max;
        qt.scale_factor[group as usize] = scale;

        for i in 0..gs {
            let quant_value = x[(group * gs + i) as usize] / scale;
            let quantized: i8 = quant_value.round() as i8;
            qt.quant_vals[(group * gs + i) as usize] = quantized.clamp(-7, 7);
        }
    }
}

#[inline(always)]
pub fn quantize(fmt: WeightFormat, dst: &mut QuantizedTensor, x: &[f32], n: usize, gs: u32) {
    match (fmt, dst) {
        (WeightFormat::Q8_0, QuantizedTensor::Q8(ref mut q)) => {
            q8_quantize(q, x, n, gs);
        }
        (WeightFormat::Q4_0, QuantizedTensor::Q4_0(ref mut q)) => {
            q4_0_quantize(q, x, n, gs);
        }
        (WeightFormat::F16, QuantizedTensor::F16(ref mut dst_vec)) => {
            dst_vec[..n].copy_from_slice(&x[..n]);
        }
        _ => panic!("quantize: format/variant mismatch"),
    }
}

#[inline(always)]
pub fn matmul(out: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize) {
    match (x, w) {
        (QuantizedTensor::Q8(ref xq), QuantizedTensor::Q8(ref wq)) => {
            matmul_q8_slices(out, &xq.quant_vals, &xq.scale_factor, &wq.quant_vals, &wq.scale_factor, n, o, gs);
        }
        (QuantizedTensor::Q4_0(ref xq), QuantizedTensor::Q4_0(ref wq)) => {
            matmul_q8_slices(out, &xq.quant_vals, &xq.scale_factor, &wq.quant_vals, &wq.scale_factor, n, o, gs);
        }
        (QuantizedTensor::F16(ref xv), QuantizedTensor::F16(ref wv)) => {
            matmul_f32(out, xv, wv, n, o);
        }
        _ => panic!("matmul: format/variant mismatch or mixed formats"),
    }
}

pub fn new_workspace_buffer(fmt: WeightFormat, len: usize, gs: usize) -> QuantizedTensor {
    match fmt {
        WeightFormat::Q8_0 => QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![0i8; len],
            scale_factor: vec![0.0f32; len / gs],
        }),
        WeightFormat::Q4_0 => QuantizedTensor::Q4_0(MutableQuantizedTensorQ4_0 {
            quant_vals: vec![0i8; len],
            scale_factor: vec![0.0f32; len / gs],
        }),
        WeightFormat::F16 => QuantizedTensor::F16(vec![0.0f32; len]),
    }
}

#[inline(always)]
pub(crate) fn matmul_q8_slices(
    xout: &mut [f32],
    x: &[i8],
    x_scale: &[f32],
    w: &[i8],
    w_scale: &[f32],
    n: usize,
    o: usize,
    gs: usize,
) {
    use rayon::prelude::*;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(row_idx, elem)| {
        let xi = row_idx * n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(col_chunk, xout_elem)| {
            let ni0 = col_chunk * 4 * n;
            let ni1 = (col_chunk * 4 + 1) * n;
            let ni2 = (col_chunk * 4 + 2) * n;
            let ni3 = (col_chunk * 4 + 3) * n;

            let mut group_start = 0;
            while group_start < n {
                let group_end = (group_start + gs).min(n);
                let remaining = group_end - group_start;
                let n_simd = remaining / 8;

                let mut ival0 = wide::i32x8::ZERO;
                let mut ival1 = wide::i32x8::ZERO;
                let mut ival2 = wide::i32x8::ZERO;
                let mut ival3 = wide::i32x8::ZERO;

                for s in 0..n_simd {
                    let base = group_start + s * 8;
                    let x_slice = &x[xi + base..xi + base + 8];
                    let w_slice0 = &w[ni0 + base..ni0 + base + 8];
                    let w_slice1 = &w[ni1 + base..ni1 + base + 8];
                    let w_slice2 = &w[ni2 + base..ni2 + base + 8];
                    let w_slice3 = &w[ni3 + base..ni3 + base + 8];

                    let x_vec = wide::i32x8::new([
                        x_slice[0] as i32, x_slice[1] as i32, x_slice[2] as i32, x_slice[3] as i32,
                        x_slice[4] as i32, x_slice[5] as i32, x_slice[6] as i32, x_slice[7] as i32,
                    ]);
                    let w_vec0 = wide::i32x8::new([
                        w_slice0[0] as i32, w_slice0[1] as i32, w_slice0[2] as i32, w_slice0[3] as i32,
                        w_slice0[4] as i32, w_slice0[5] as i32, w_slice0[6] as i32, w_slice0[7] as i32,
                    ]);
                    let w_vec1 = wide::i32x8::new([
                        w_slice1[0] as i32, w_slice1[1] as i32, w_slice1[2] as i32, w_slice1[3] as i32,
                        w_slice1[4] as i32, w_slice1[5] as i32, w_slice1[6] as i32, w_slice1[7] as i32,
                    ]);
                    let w_vec2 = wide::i32x8::new([
                        w_slice2[0] as i32, w_slice2[1] as i32, w_slice2[2] as i32, w_slice2[3] as i32,
                        w_slice2[4] as i32, w_slice2[5] as i32, w_slice2[6] as i32, w_slice2[7] as i32,
                    ]);
                    let w_vec3 = wide::i32x8::new([
                        w_slice3[0] as i32, w_slice3[1] as i32, w_slice3[2] as i32, w_slice3[3] as i32,
                        w_slice3[4] as i32, w_slice3[5] as i32, w_slice3[6] as i32, w_slice3[7] as i32,
                    ]);

                    ival0 += x_vec * w_vec0;
                    ival1 += x_vec * w_vec1;
                    ival2 += x_vec * w_vec2;
                    ival3 += x_vec * w_vec3;
                }

                xout_elem[0] += (ival0.reduce_add() as f32)
                    * w_scale[(ni0 + group_start) / gs]
                    * x_scale[(xi + group_start) / gs];
                xout_elem[1] += (ival1.reduce_add() as f32)
                    * w_scale[(ni1 + group_start) / gs]
                    * x_scale[(xi + group_start) / gs];
                xout_elem[2] += (ival2.reduce_add() as f32)
                    * w_scale[(ni2 + group_start) / gs]
                    * x_scale[(xi + group_start) / gs];
                xout_elem[3] += (ival3.reduce_add() as f32)
                    * w_scale[(ni3 + group_start) / gs]
                    * x_scale[(xi + group_start) / gs];

                group_start = group_end;
            }
        });

        let handled = (o / 4) * 4;
        for col in handled..o {
            let ni = col * n;
            let mut sum = 0.0f32;
            let mut group_start = 0;
            while group_start < n {
                let group_end = (group_start + gs).min(n);
                let mut block_sum = 0i32;
                for k in group_start..group_end {
                    block_sum += x[xi + k] as i32 * w[ni + k] as i32;
                }
                sum += block_sum as f32
                    * w_scale[(ni + group_start) / gs]
                    * x_scale[(xi + group_start) / gs];
                group_start = group_end;
            }
            elem[col] = sum;
        }
    });
}

#[inline(always)]
pub(crate) fn matmul_f32(
    xout: &mut [f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    o: usize,
) {
    use rayon::prelude::*;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(row_idx, elem)| {
        let xi = row_idx * n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(col_chunk, xout_elem)| {
            let w_base = col_chunk * 4;
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            for i in 0..n {
                let xv = x[xi + i];
                sum0 += xv * w[(w_base + 0) * n + i];
                sum1 += xv * w[(w_base + 1) * n + i];
                sum2 += xv * w[(w_base + 2) * n + i];
                sum3 += xv * w[(w_base + 3) * n + i];
            }
            xout_elem[0] = sum0;
            xout_elem[1] = sum1;
            xout_elem[2] = sum2;
            xout_elem[3] = sum3;
        });

        let handled = (o / 4) * 4;
        for col in handled..o {
            let w_base = col;
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += x[xi + i] * w[w_base * n + i];
            }
            elem[col] = sum;
        }
    });
}
