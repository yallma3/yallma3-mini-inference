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


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QuantType {
    None = 0,
    Q8_0 = 1,
    Q4_0 = 2,
    F16 = 3,
}

#[derive(Debug)]
pub struct MutableQuantizedTensorQ8 {
    pub quant_vals: Vec<i8>,
    pub scale_factor: Vec<f32>,
}

pub fn quantize(qx: &mut MutableQuantizedTensorQ8, x: &[f32], n: usize, gs: u32) {
    let num_groups: u32 = n as u32 / gs;
    let q_max: f32 = 127.0f32;

    for group in 0..num_groups {
        let mut wmax: f32 = 0.0;
        for i in 0..gs {
            let val: f32 = x[(group * gs + i) as usize].abs();
            if val > wmax {
                wmax = val;
            }
        }

        let scale = wmax / q_max;

        qx.scale_factor[group as usize] = scale;

        for i in 0..gs {
            let quant_value = x[(group * gs + i) as usize] / scale;
            let quantized: i8 = quant_value.round() as i8;
            qx.quant_vals[(group * gs + i) as usize] = quantized;
        }
    }
}

