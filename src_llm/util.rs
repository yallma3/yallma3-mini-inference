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


use crate::base_transformer::TransformerShape;
use crate::quantization::MutableQuantizedTensorQ8;

#[cfg(feature = "debug_prints")]
use hxd::{AsHexd, IntoFallibleHexd};

#[cfg(feature = "debug_prints")]
#[derive(Clone, Copy)]
pub enum PrintRange {
    Head,
    Tail,
    HeadNTail,
}

#[cfg(feature = "debug_prints")]
#[derive(Clone, Copy)]
pub enum PrintFormat {
    Full,
    Summary1,
}

#[cfg(feature = "debug_prints")]
#[derive(Clone, Copy)]
pub enum MatrixPrecision {
    F32,
    F16,
    I16,
    U16,
    I8,
    U8,
}

#[cfg(feature = "debug_prints")]
impl MatrixPrecision {
    fn size(&self) -> usize {
        match self {
            MatrixPrecision::F32 => 4,
            MatrixPrecision::F16 | MatrixPrecision::I16 | MatrixPrecision::U16 => 2,
            MatrixPrecision::I8 | MatrixPrecision::U8 => 1,
        }
    }
}

#[cfg(feature = "debug_prints")]
pub fn print_matrix_2d(
    title: &str,
    data: &[u8],
    precision: MatrixPrecision,
    rows: usize,
    cols: usize,
    format: Option<PrintFormat>,
) {
    let format = format.unwrap_or(PrintFormat::Full);
    if !title.is_empty() {
        println!("=== {} ({}x{}) ===", title, rows, cols);
    }
    let elem_size = precision.size();

    let col_indices: Vec<usize> = match format {
        PrintFormat::Full => (0..cols).collect(),
        PrintFormat::Summary1 if cols <= 16 => (0..cols).collect(),
        PrintFormat::Summary1 => {
            let mut indices: Vec<usize> = (0..8).collect();
            indices.push(999);
            indices.extend((cols - 8)..cols);
            indices
        }
    };

    let row_indices: Vec<usize> = match format {
        PrintFormat::Full => (0..rows).collect(),
        PrintFormat::Summary1 if rows <= 8 => (0..rows).collect(),
        PrintFormat::Summary1 => {
            let mut indices: Vec<usize> = (0..4).collect();
            indices.push(999);
            indices.extend((rows - 4)..rows);
            indices
        }
    };

    println!(
        "       {}",
        col_indices
            .iter()
            .map(|&i| {
                if i == 999 {
                    "...".to_string()
                } else {
                    format!("{:4}", i)
                }
            })
            .collect::<String>()
    );
    for &r in &row_indices {
        let mut row_str = if r == 999 {
            " ... ".to_string()
        } else {
            format!("{:4}: ", r)
        };
        for &c in &col_indices {
            if c == 999 {
                row_str.push_str(&format!("{:>8}", "..."));
                continue;
            }
            let idx = (r * cols + c) * elem_size;
            if idx + elem_size > data.len() {
                row_str.push_str("   ???");
                continue;
            }
            let val = match precision {
                MatrixPrecision::F32 => {
                    let bytes: [u8; 4] = [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]];
                    f32::from_le_bytes(bytes).to_string()
                }
                MatrixPrecision::F16 => {
                    let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                    let val = half::f16::from_le_bytes(bytes);
                    format!("{:.4}", val)
                }
                MatrixPrecision::I16 => {
                    let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                    let val = i16::from_le_bytes(bytes);
                    val.to_string()
                }
                MatrixPrecision::U16 => {
                    let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                    let val = u16::from_le_bytes(bytes);
                    val.to_string()
                }
                MatrixPrecision::I8 => (data[idx] as i8).to_string(),
                MatrixPrecision::U8 => data[idx].to_string(),
            };
            row_str.push_str(&format!("{:>8}", val));
        }
        println!("{}", row_str);
    }
    println!("");
}

#[cfg(feature = "debug_prints")]
pub fn print_matrix_3d(
    title: &str,
    data: &[u8],
    precision: MatrixPrecision,
    outer_rows: usize,
    inner_rows: usize,
    inner_cols: usize,
    format: Option<PrintFormat>,
) {
    let format = format.unwrap_or(PrintFormat::Full);
    let indent = "  ";
    if !title.is_empty() {
        println!(
            "=== {} ({}x{}x{}) ===",
            title, outer_rows, inner_rows, inner_cols
        );
    }
    let elem_size = precision.size();

    let col_indices: Vec<usize> = match format {
        PrintFormat::Full => (0..inner_cols).collect(),
        PrintFormat::Summary1 if inner_cols <= 16 => (0..inner_cols).collect(),
        PrintFormat::Summary1 => {
            let mut indices: Vec<usize> = (0..8).collect();
            indices.push(999);
            indices.extend((inner_cols - 8)..inner_cols);
            indices
        }
    };

    let row_indices: Vec<usize> = match format {
        PrintFormat::Full => (0..inner_rows).collect(),
        PrintFormat::Summary1 if inner_rows <= 8 => (0..inner_rows).collect(),
        PrintFormat::Summary1 => {
            let mut indices: Vec<usize> = (0..4).collect();
            indices.push(999);
            indices.extend((inner_rows - 4)..inner_rows);
            indices
        }
    };

    let outer_row_indices: Vec<usize> = match format {
        PrintFormat::Full => (0..outer_rows).collect(),
        PrintFormat::Summary1 if outer_rows <= 8 => (0..outer_rows).collect(),
        PrintFormat::Summary1 => {
            let mut indices: Vec<usize> = (0..4).collect();
            indices.push(999);
            indices.extend((outer_rows - 4)..outer_rows);
            indices
        }
    };

    for &or in &outer_row_indices {
        if or == 999 {
            println!("{}... {} outer rows omitted ...", indent, outer_rows - 8);
            continue;
        }
        let header = col_indices
            .iter()
            .map(|&i| {
                if i == 999 {
                    "...".to_string()
                } else {
                    format!("{:>8}", i)
                }
            })
            .collect::<String>();
        println!("{}Outer row {}:{:>8}", indent, or, " ");
        println!("{}{:>8}{}", indent, " ", header);
        for &r in &row_indices {
            let mut row_str = if r == 999 {
                format!("{}{:>8}: ", indent, "")
            } else {
                format!("{}{:>8}: ", indent, r)
            };
            for &c in &col_indices {
                if c == 999 {
                    row_str.push_str(&format!("{:>8}", "..."));
                    continue;
                }
                let idx = (or * inner_rows * inner_cols + r * inner_cols + c) * elem_size;
                if idx + elem_size > data.len() {
                    row_str.push_str(&format!("{:>8}", "???"));
                    continue;
                }
                let val = match precision {
                    MatrixPrecision::F32 => {
                        let bytes: [u8; 4] =
                            [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]];
                        let val = f32::from_le_bytes(bytes);
                        if matches!(format, PrintFormat::Summary1) {
                            format!("{:.5}", val)
                        } else {
                            val.to_string()
                        }
                    }
                    MatrixPrecision::F16 => {
                        let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                        let val = half::f16::from_le_bytes(bytes);
                        format!("{:.4}", val)
                    }
                    MatrixPrecision::I16 => {
                        let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                        let val = i16::from_le_bytes(bytes);
                        val.to_string()
                    }
                    MatrixPrecision::U16 => {
                        let bytes: [u8; 2] = [data[idx], data[idx + 1]];
                        let val = u16::from_le_bytes(bytes);
                        val.to_string()
                    }
                    MatrixPrecision::I8 => (data[idx] as i8).to_string(),
                    MatrixPrecision::U8 => data[idx].to_string(),
                };
                row_str.push_str(&format!("{:>8}", val));
            }
            println!("{}", row_str);
        }
    }
    println!("");
}

#[cfg(feature = "debug_prints")]
pub fn print_float_matrix_3d(
    title: &str,
    floats: &[f32],
    outer_rows: usize,
    inner_rows: usize,
    inner_cols: usize,
    format: Option<PrintFormat>,
) {
    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    print_matrix_3d(
        title,
        &bytes,
        MatrixPrecision::F32,
        outer_rows,
        inner_rows,
        inner_cols,
        format,
    );
}

#[cfg(feature = "debug_prints")]
pub fn print_byte_mem(title: &str, bytes: &[u8], range: Option<(PrintRange, usize)>) {
    let (range_variant, slice_len) = range.unwrap_or((PrintRange::Head, bytes.len()));
    if !title.is_empty() {
        println!("=== {} ===", title);
    }
    match range_variant {
        PrintRange::Head => {
            if !title.is_empty() && slice_len < bytes.len() {
                println!(" === showing top {} out of {} ==", slice_len, bytes.len());
            }
            bytes[..slice_len].hexd().dump();
        }
        PrintRange::Tail => {
            if !title.is_empty() && slice_len < bytes.len() {
                println!(
                    " === showing bottom {} out of {} ==",
                    slice_len,
                    bytes.len()
                );
            }
            bytes[bytes.len() - slice_len..].hexd().dump();
        }
        PrintRange::HeadNTail => {
            if !title.is_empty() && slice_len < bytes.len() / 2 {
                println!(
                    " === showing top {} and bottom {} out of {} ==",
                    slice_len,
                    slice_len,
                    bytes.len()
                );
            }
            bytes[..slice_len].hexd().dump();
            println!("... {} bytes omitted ...", bytes.len() - slice_len * 2);
            bytes[bytes.len() - slice_len..].hexd().dump();
        }
    }
    println!("");
}

#[cfg(feature = "debug_prints")]
pub fn print_float_mem(title: &str, floats: &[f32], range: Option<(PrintRange, usize)>) {
    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let byte_range = range.map(|(r, n)| (r, n * 4));
    print_byte_mem(title, &bytes, byte_range);
}

#[cfg(target_arch = "x86_64")]
use wide::f32x8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Avx512,
    Avx2,
    Scalar,
}

fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

static SIMD_LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

pub fn get_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(detect_simd_level)
}

pub fn slice_to_u32(slice: &[u8]) -> u32 {
    let bytes: [u8; 4] = slice
        .try_into()
        .expect("Slice must be exactly 4 bytes long");
    u32::from_le_bytes(bytes)
}

pub fn slice_to_f32(slice: &[u8]) -> f32 {
    let bytes: [u8; 4] = slice
        .try_into()
        .expect("Slice must be exactly 4 bytes long");
    f32::from_le_bytes(bytes)
}

//LMRS Version used for comparison only
pub fn rmsnorm(
    o: &mut [f32],
    x: &[f32],
    weight: &[f32],
    size: usize,
    eps: f32,
    add_unit_offset: bool,
) {
    let n_simd = size / 8;

    let mut ss_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        ss_sim += x_vec * x_vec;
    }

    let mut ss = ss_sim.reduce_add();

    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);

        let r = if add_unit_offset {
            ((1.0 + w_vec) * (ss * x_vec)).to_array()
        } else {
            (w_vec * (ss * x_vec)).to_array()
        };

        for k in 0..8 {
            o[(j * 8) + k] = r[k];
        }
    }
}

pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32, add_unit_offset: bool) {
    let n = x.len() as f32;
    let mut sum_sq = 0.0;
    for &val in x.iter() {
        sum_sq += val * val;
    }
    let rms = (sum_sq / n + eps).sqrt();

    for (x_val, w_val) in x.iter_mut().zip(weight.iter()) {
        *x_val = (*x_val / rms)
            * (if add_unit_offset {
                1.0 + *w_val
            } else {
                *w_val
            });
    }
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = *val / (1.0 + (-*val).exp());
    }
}

pub fn load_q8_tensor_st(
    data: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
    weight_start: usize,
    scale_start: usize,
    bias_start: usize,
) -> MutableQuantizedTensorQ8 {
    let n = rows * cols;
    let groups = n / group_size;
    let packed_cols = cols / 4;

    // Step 1: unpack i8 values
    let mut vals = vec![0i8; n];
    for r in 0..rows {
        for pc in 0..packed_cols {
            let off = weight_start + (r * packed_cols + pc) * 4;
            let packed = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let dst = r * cols + pc * 4;
            vals[dst + 0] = (packed >> 0) as u8 as i8;
            vals[dst + 1] = (packed >> 8) as u8 as i8;
            vals[dst + 2] = (packed >> 16) as u8 as i8;
            vals[dst + 3] = (packed >> 24) as u8 as i8;
        }
    }

    // Step 2: read f16 scales & biases, then dequantize → requantize symmetric per group
    let mut qt = MutableQuantizedTensorQ8 {
        quant_vals: vec![0i8; n],
        scale_factor: vec![0.0f32; groups],
    };

    for g in 0..groups {
        let s_bits = u16::from_le_bytes(data[scale_start + g * 2..][..2].try_into().unwrap());
        let scale = half::f16::from_bits(s_bits).to_f32();
        let b_bits = u16::from_le_bytes(data[bias_start + g * 2..][..2].try_into().unwrap());
        let bias = half::f16::from_bits(b_bits).to_f32();

        let start = g * group_size;
        let end = (start + group_size).min(n);

        // dequantize
        let mut max_abs = 0.0f32;
        let mut deq = [0.0f32; 256];
        for i in start..end {
            let f = (vals[i] as u8) as f32 * scale + bias;
            deq[i - start] = f;
            let a = f.abs();
            if a > max_abs { max_abs = a; }
        }

        // symmetric requantize
        let sym_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        qt.scale_factor[g] = sym_scale;

        for i in start..end {
            let q = (deq[i - start] / sym_scale).round() as i32;
            qt.quant_vals[i] = q.clamp(-128, 127) as i8;
        }
    }

    qt
}

pub fn load_q8_tensor_st_asym(
    data: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
    weight_start: usize,
    scale_start: usize,
    bias_start: usize,
) -> (MutableQuantizedTensorQ8, Vec<f32>) {
    let n = rows * cols;
    let groups = n / group_size;
    let packed_cols = cols / 4;

    let mut qt = MutableQuantizedTensorQ8 {
        quant_vals: vec![0i8; n],
        scale_factor: vec![0.0f32; groups],
    };

    let mut bias = vec![0.0f32; groups];

    for r in 0..rows {
        for pc in 0..packed_cols {
            let off = weight_start + (r * packed_cols + pc) * 4;
            let packed = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let dst = r * cols + pc * 4;
            qt.quant_vals[dst + 0] = (packed >> 0) as u8 as i8;
            qt.quant_vals[dst + 1] = (packed >> 8) as u8 as i8;
            qt.quant_vals[dst + 2] = (packed >> 16) as u8 as i8;
            qt.quant_vals[dst + 3] = (packed >> 24) as u8 as i8;
        }
    }

    for g in 0..groups {
        let s_bits = u16::from_le_bytes(data[scale_start + g * 2..][..2].try_into().unwrap());
        qt.scale_factor[g] = half::f16::from_bits(s_bits).to_f32();
        let b_bits = u16::from_le_bytes(data[bias_start + g * 2..][..2].try_into().unwrap());
        bias[g] = half::f16::from_bits(b_bits).to_f32();
    }

    (qt, bias)
}

pub fn load_q8_tenstor_lmrs(
    raw_model: &[u8],
    offset: usize,
    vocab_size: usize,
    dim: usize,
    group_size: usize,
) -> (MutableQuantizedTensorQ8, usize) {
    let mem_size = vocab_size * dim;
    let groups = mem_size / group_size;
    let mut qt = MutableQuantizedTensorQ8 {
        quant_vals: vec![0; mem_size],
        scale_factor: vec![0.0; groups],
    };
    qt.quant_vals.copy_from_slice(unsafe {
        std::slice::from_raw_parts(
            raw_model[offset..offset + mem_size].as_ptr() as *const i8,
            mem_size,
        )
    });
    let scale_offset = offset + mem_size;
    qt.scale_factor.copy_from_slice(unsafe {
        std::slice::from_raw_parts(
            raw_model[scale_offset..scale_offset + (groups * size_of::<f32>())].as_ptr() as *const f32,
            groups,
        )
    });
    let total_size = mem_size + groups * size_of::<f32>();
    (qt, total_size)
}

pub fn rope(shape: &TransformerShape, sq: &mut [f32], k: &mut [f32], pos: u32) {
    let rope_factor = 32.0;
    let low_freq_factor = 1.0;
    let high_freq_factor = 4.0;
    let old_context_len = shape.ctx_len as f32;
    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;
    let rope_theta = shape.rope_theta;
    let head_size = shape.head_size as usize;
    let half_head_size = head_size / 2;
    let n_heads = shape.n_heads as usize;
    let n_kv_heads = shape.n_kv_heads as usize;

    for j in 0..half_head_size {
        let head_dim = j * 2;
        let mut freq = rope_theta.powf(-(head_dim as f32) / head_size as f32);

        let wavelen = (2.0 * std::f32::consts::PI) / freq;
        if wavelen > low_freq_wavelen {
            freq /= rope_factor;
        } else if wavelen <= low_freq_wavelen && wavelen >= high_freq_wavelen {
            let smooth_factor = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            freq = (1.0 - smooth_factor) * freq / rope_factor + smooth_factor * freq;
        }

        let val = (pos as f32) * freq;
        let fcr = val.cos();
        let fci = val.sin();

        for attention_head in 0..n_heads {
            let offset = attention_head * head_size + j;
            let v0q = sq[offset];
            let v1q = sq[offset + half_head_size];
            sq[offset] = v0q * fcr - v1q * fci;
            sq[offset + half_head_size] = v0q * fci + v1q * fcr;
        }

        for kv_idx in 0..n_kv_heads {
            let offset = kv_idx * head_size + j;
            let v0k = k[offset];
            let v1k = k[offset + half_head_size];
            k[offset] = v0k * fcr - v1k * fci;
            k[offset + half_head_size] = v0k * fci + v1k * fcr;
        }
    }
}

/// GGUF-specific RoPE with consecutive-element pairing (LLAMA_ROPE_TYPE_NORM).
///
/// The GGUF converter (convert_hf_to_gguf.py) permutes Q/K weight matrices from
/// HuggingFace layout to an interleaved layout:
///   HF order:       [d0, d1, d2, ..., d_{d/2-1}, d_{d/2}, d_{d/2+1}, ..., d_{d-1}]
///   Permuted order: [d0, d_{d/2}, d1, d_{d/2+1}, d2, d_{d/2+2}, ...]
///
/// This function pairs consecutive elements (2k, 2k+1) — which after the permutation
/// becomes mathematically equivalent to first-half-second-half RoPE on the original
/// HuggingFace layout. Both pipelines produce identical numerical results:
///
///   SafeTensor:  unpermuted Q weights + rope() (first-half-second-half)
///   GGUF:        permuted   Q weights + rope_gguf() (consecutive-pairs)  ✓ same
///
/// Supports full YaRN scaling (freq_scale, ext_factor, attn_factor, corr_dims)
/// matching llama.cpp's ggml_compute_forward_rope with LLAMA_ROPE_TYPE_NORM.
pub fn rope_gguf(shape: &TransformerShape, sq: &mut [f32], k: &mut [f32], pos: u32) {
    let rope_theta = shape.rope_theta;
    let head_size = shape.head_size as usize;
    let n_heads = shape.n_heads as usize;
    let n_kv_heads = shape.n_kv_heads as usize;
    let freq_scale = shape.rope_freq_scale;
    let ext_factor = shape.rope_ext_factor;
    let attn_factor = shape.rope_attn_factor;
    let n_ctx_orig = shape.rope_original_ctx_len as f32;

    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;

    let n_pairs = head_size / 2;
    let theta_scale = rope_theta.powf(-2.0 / head_size as f32);

    let corr_start = (n_ctx_orig / (beta_fast * 2.0 * std::f32::consts::PI))
        .ln()
        .mul_add(head_size as f32, 0.0)
        / (2.0 * rope_theta.ln());
    let corr_start = corr_start.floor().max(0.0);

    let corr_end = (n_ctx_orig / (beta_slow * 2.0 * std::f32::consts::PI))
        .ln()
        .mul_add(head_size as f32, 0.0)
        / (2.0 * rope_theta.ln());

    let corr_dims_1 = corr_start.min(corr_end)
        + corr_end.min(n_pairs as f32)
        - corr_start.max(0.0);

    let mut theta = pos as f32;
    for k_pair in 0..n_pairs {
        let ff = 1.0f32;

        let theta_interp = freq_scale * theta / ff;
        let mut t = theta_interp;
        if ext_factor != 0.0 {
            let i0 = (k_pair * 2) as f32;
            let ramp_mix = rope_yarn_ramp(corr_start, corr_dims_1, i0) * ext_factor;
            t = theta_interp * (1.0 - ramp_mix) + theta / ff * ramp_mix;
        }

        let cos_theta = t.cos() * attn_factor;
        let sin_theta = t.sin() * attn_factor;

        let col0 = k_pair * 2;

        for head in 0..n_heads {
            let off = head * head_size + col0;
            let v0 = sq[off];
            let v1 = sq[off + 1];
            sq[off] = v0 * cos_theta - v1 * sin_theta;
            sq[off + 1] = v0 * sin_theta + v1 * cos_theta;
        }
        for kv_head in 0..n_kv_heads {
            let off = kv_head * head_size + col0;
            let v0 = k[off];
            let v1 = k[off + 1];
            k[off] = v0 * cos_theta - v1 * sin_theta;
            k[off + 1] = v0 * sin_theta + v1 * cos_theta;
        }

        theta *= theta_scale;
    }
}

#[inline]
fn rope_yarn_ramp(low: f32, high: f32, i0: f32) -> f32 {
    let y = (i0 * 0.5 - low) / (high - low).max(0.001);
    1.0 - (1.0f32).min(y.max(0.0))
}
