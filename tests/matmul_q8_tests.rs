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


use yallma3_llm::quantization::MutableQuantizedTensorQ8;
use yallma3_llm::weight_type::QuantizedTensor;

fn matmul_q8(xout: &mut [f32], x: &MutableQuantizedTensorQ8, w: &MutableQuantizedTensorQ8, n: usize, o: usize, gs: usize) {
    let xq = QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
        quant_vals: x.quant_vals.clone(),
        scale_factor: x.scale_factor.clone(),
    });
    let wq = QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
        quant_vals: w.quant_vals.clone(),
        scale_factor: w.scale_factor.clone(),
    });
    yallma3_llm::weight_type::matmul(xout, &xq, &wq, n, o, gs);
}

#[test]
fn test_matmul_q8_remainder_columns() {
    let batch = 1;
    let n = 64;
    let o = 33;
    let gs = 32;

    let x_q = MutableQuantizedTensorQ8 {
        quant_vals: vec![1i8; batch * n],
        scale_factor: vec![1.0f32; (batch * n) / gs],
    };
    let w_q = MutableQuantizedTensorQ8 {
        quant_vals: vec![1i8; o * n],
        scale_factor: vec![1.0f32; (o * n) / gs],
    };

    let mut out = vec![0.0f32; batch * o];
    matmul_q8(&mut out, &x_q, &w_q, n, o, gs);

    for i in 0..o {
        let diff = (out[i] - n as f32).abs();
        assert!(diff < 1e-5, "out[{}] = {}, expected {}", i, out[i], n);
    }
}
