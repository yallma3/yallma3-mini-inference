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


//! Tests comparing forward_sequential vs forward for correctness.
//!
//! Run with: `cargo test forward_compare --test forward_compare -- --nocapture`

use yallma3_llm::base_transformer::{
    InferenceSession, ModelFamily, ModelWeights, TransformerBase, TransformerShape,
};
use yallma3_llm::quantization::{MutableQuantizedTensorQ8, QuantType};
use yallma3_llm::weight_type::QuantizedTensor;

const DIM: u32 = 32;
const HEAD_SIZE: u32 = 16;
const N_HEADS: u32 = 2;
const N_KV_HEADS: u32 = 2;
const VOCAB_SIZE: u32 = 10;
const N_LAYERS: u32 = 1;
const CTX_LEN: u32 = 4;
const GROUP_SIZE: u32 = 8;

struct MockTransformer {
    base: TransformerBase,
}

fn simple_rms_norm(x: &mut [f32], weight: &[f32], _eps: f32) {
    let n = x.len();
    let mut ms = 0.0f32;
    for i in 0..n {
        ms += x[i] * x[i];
    }
    ms = (ms / n as f32).sqrt();
    for i in 0..n {
        let norm = x[i] / (ms + 1e-5);
        x[i] = (1.0 + weight[i] * 0.1) * norm;
    }
}

fn simple_quantize(qx: &mut MutableQuantizedTensorQ8, x: &[f32], n: usize, gs: usize) {
    let num_groups = n / gs;
    let q_max: f32 = 127.0f32;

    for group in 0..num_groups {
        let mut wmax: f32 = 0.0;
        for i in 0..gs {
            let val = x[group * gs + i].abs();
            if val > wmax {
                wmax = val;
            }
        }

        let scale = wmax / q_max;
        qx.scale_factor[group] = scale;

        for i in 0..gs {
            let quantized = x[group * gs + i] / scale;
            qx.quant_vals[group * gs + i] = quantized.round() as i8;
        }
    }
}

fn simple_dequantize_mat_mul(
    output: &mut [f32],
    input: &MutableQuantizedTensorQ8,
    weights: &QuantizedTensor,
    group_size: usize,
) {
    let w = match weights {
        QuantizedTensor::Q8(ref q) => q,
        _ => panic!("forward_compare: only Q8_0 format is supported"),
    };
    let in_dim = input.quant_vals.len();
    let out_dim = output.len();
    let n_groups_per_out = in_dim / group_size;

    for i in 0..out_dim {
        let mut sum = 0.0f32;
        let weight_offset = i * in_dim;

        for j in 0..in_dim {
            let group = j / group_size;
            let w_val = w.quant_vals[weight_offset + j] as f32;
            let scale = w.scale_factor[i * n_groups_per_out + group];
            let i_scale = input.scale_factor[group];
            sum += input.quant_vals[j] as f32 * i_scale * w_val * scale;
        }
        output[i] = sum;
    }
}

impl MockTransformer {
    fn new() -> Self {
        let shape = TransformerShape {
            dimension: DIM,
            hidden_dimension: DIM * 4,
            n_heads: N_HEADS,
            head_size: HEAD_SIZE,
            n_kv_heads: N_KV_HEADS,
            vocab_size: VOCAB_SIZE,
            n_layers: N_LAYERS,
            ctx_len: CTX_LEN,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            q_type: QuantType::Q8_0,
            group_size: GROUP_SIZE,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            rope_original_ctx_len: CTX_LEN,
            attention_scale: 0.0,
        };

        let kv_dim = (N_KV_HEADS * HEAD_SIZE) as usize;
        let attention_dim = (N_HEADS * HEAD_SIZE) as usize;
        let dim = DIM as usize;

        let mut token_embedding = vec![0.0f32; dim * VOCAB_SIZE as usize];
        for i in 0..token_embedding.len() {
            token_embedding[i] = ((i as f32) % 10.0) * 0.1;
        }

        let att = vec![vec![1.0f32; dim]; N_LAYERS as usize];

        let wq = vec![QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![1i8; attention_dim * dim],
            scale_factor: vec![0.1f32; (attention_dim * dim) / GROUP_SIZE as usize],
        })];
        let wk = vec![QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![1i8; kv_dim * dim],
            scale_factor: vec![0.1f32; (kv_dim * dim) / GROUP_SIZE as usize],
        })];
        let wv = vec![QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![1i8; kv_dim * dim],
            scale_factor: vec![0.1f32; (kv_dim * dim) / GROUP_SIZE as usize],
        })];

        let wo = vec![QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![1i8; attention_dim * dim],
            scale_factor: vec![0.1f32; (attention_dim * dim) / GROUP_SIZE as usize],
        })];

        let w_rms_final = vec![1.0f32; dim];

        let weights = Some(ModelWeights {
            token_embedding,
            token_embedding_quant: None,
            wq,
            wk,
            wv,
            wo,
            w_rms_att: att,
            w1: vec![],
            w2: vec![],
            w3: vec![],
            w_rms_post_att: vec![],
            w_rms_final,
        });

        MockTransformer {
            base: TransformerBase {
                model_info: "Mock".to_string(),
                model_family: Some(ModelFamily::Llama),
                shape: Some(shape),
                weights,
                tokenizer: None,
            },
        }
    }
}

impl MockTransformer {
    fn _forward_impl(
        &self,
        token: u32,
        pos: u32,
        _is_generation: bool,
        sequential: bool,
        session: &mut InferenceSession,
    ) -> Vec<f32> {
        let shape = self.base.shape.as_ref().expect("Shape not initialized");
        let weights = self.base.weights.as_ref().expect("Weights not initialized");

        let dim = DIM as usize;
        let mut x = vec![0.0; dim];

        let token_embed_idx = (token as usize) * dim;
        if token_embed_idx + dim <= weights.token_embedding.len() {
            x.copy_from_slice(&weights.token_embedding[token_embed_idx..token_embed_idx + dim]);
        }

        for layer in 0..shape.n_layers as usize {
            let current_token_embeddings = &mut x.to_vec();

            simple_rms_norm(
                current_token_embeddings,
                &weights.w_rms_att[layer],
                shape.rms_norm_eps,
            );

            let attention_dim = (shape.n_heads * shape.head_size) as usize;
            let kv_dim = (shape.n_kv_heads * shape.head_size) as usize;

            let mut sq = vec![0.0f32; attention_dim];

            let layer_cache_offset = layer * (shape.ctx_len as usize) * kv_dim;
            let pos_offset = (pos as usize) * kv_dim;

            let k = &mut session.key_cache
                [layer_cache_offset + pos_offset..layer_cache_offset + pos_offset + kv_dim];
            let v = &mut session.value_cache
                [layer_cache_offset + pos_offset..layer_cache_offset + pos_offset + kv_dim];

            let quant_current_token = &mut MutableQuantizedTensorQ8 {
                quant_vals: vec![0; dim],
                scale_factor: vec![0.0; dim / GROUP_SIZE as usize],
            };

            simple_quantize(
                quant_current_token,
                &current_token_embeddings,
                dim,
                shape.group_size as usize,
            );

            simple_dequantize_mat_mul(
                &mut sq,
                quant_current_token,
                &weights.wq[layer],
                shape.group_size as usize,
            );

            simple_dequantize_mat_mul(
                k,
                quant_current_token,
                &weights.wk[layer],
                shape.group_size as usize,
            );

            simple_dequantize_mat_mul(
                v,
                quant_current_token,
                &weights.wv[layer],
                shape.group_size as usize,
            );

            let rope_theta = shape.rope_theta;
            let head_size = shape.head_size as usize;
            let half_head_size = head_size / 2;
            let n_heads = shape.n_heads as usize;
            let n_kv_heads = shape.n_kv_heads as usize;

            for j in 0..half_head_size {
                let head_dim = j * 2;
                let freq = rope_theta.powf(-(head_dim as f32) / head_size as f32);
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

            let q_per_kv = shape.n_heads as usize / shape.n_kv_heads as usize;
            let head_size = shape.head_size as usize;
            let ctx_len = shape.ctx_len as usize;
            let layer_cache_offset = layer * (ctx_len * kv_dim);

            const CAUSAL_MASK_VALUE: f32 = -1e10;
            let scaling = 1.0 / (head_size as f32).sqrt();

            let mut att_output = vec![0.0f32; attention_dim];

            let max_pos = if sequential {
                pos as usize
            } else {
                pos as usize
            };

            for head in 0..shape.n_heads as usize {
                let q_offset = head * head_size;
                let kv_head = head / q_per_kv;

                let mut scores = vec![0.0f32; max_pos];
                let mut max_score = CAUSAL_MASK_VALUE;

                for t in 0..max_pos {
                    let mut score = 0.0f32;
                    for d in 0..head_size {
                        let q_val = sq[q_offset + d];
                        let key_pos_offset = layer_cache_offset + t * kv_dim + kv_head * head_size;
                        let k_val = session.key_cache[key_pos_offset + d];
                        score += q_val * k_val;
                    }
                    score *= scaling;
                    scores[t] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for t in 0..max_pos {
                    scores[t] = (scores[t] - max_score).exp();
                    sum_exp += scores[t];
                }
                let softmax_scale = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                for t in 0..max_pos {
                    scores[t] *= softmax_scale;
                }

                for d in 0..head_size {
                    let mut output_val = 0.0f32;
                    for t in 0..max_pos {
                        let value_pos_offset =
                            layer_cache_offset + t * kv_dim + kv_head * head_size;
                        let v_val = session.value_cache[value_pos_offset + d];
                        output_val += scores[t] * v_val;
                    }
                    att_output[q_offset + d] = output_val;
                }
            }

            let dim = shape.dimension as usize;
            let group_size = shape.group_size as usize;

            for i in 0..dim {
                let mut sum = 0.0f32;
                let wo = weights.wo[layer].as_q8();
                for j in 0..attention_dim {
                    let w_val = wo.quant_vals[j * dim + i] as f32;
                    let scale = wo.scale_factor[j / group_size];
                    sum += att_output[j] * w_val * scale;
                }
                current_token_embeddings[i] = sum;
            }
        }

        simple_rms_norm(&mut x, &weights.w_rms_final, shape.rms_norm_eps);

        let vocab_size = shape.vocab_size as usize;
        let mut logits = vec![0.0f32; vocab_size];
        let embedding_dim = weights.token_embedding.len() / vocab_size;
        for i in 0..vocab_size {
            let mut sum = 0.0f32;
            for j in 0..embedding_dim {
                sum += x[j] * weights.token_embedding[i * embedding_dim + j];
            }
            logits[i] = sum;
        }

        logits
    }
}

#[test]
fn test_forward_sequential_single_token() {
    let transformer = MockTransformer::new();
    let mut session = InferenceSession::new(transformer.base.shape.as_ref().unwrap());
    let token = 1u32;
    let pos = 0u32;

    let logits = transformer._forward_impl(token, pos, false, true, &mut session);

    assert_eq!(logits.len(), VOCAB_SIZE as usize);
    for l in &logits {
        assert!(l.is_finite(), "Logits should be finite");
    }
}

#[test]
fn test_forward_parallel_single_token() {
    let transformer = MockTransformer::new();
    let mut session = InferenceSession::new(transformer.base.shape.as_ref().unwrap());
    let token = 1u32;
    let pos = 0u32;

    let logits = transformer._forward_impl(token, pos, false, false, &mut session);

    assert_eq!(logits.len(), VOCAB_SIZE as usize);
    for l in &logits {
        assert!(l.is_finite(), "Logits should be finite");
    }
}

#[test]
fn test_forward_compare_at_pos_zero() {
    let t1 = MockTransformer::new();
    let t2 = MockTransformer::new();
    let mut s1 = InferenceSession::new(t1.base.shape.as_ref().unwrap());
    let mut s2 = InferenceSession::new(t2.base.shape.as_ref().unwrap());

    let token = 1u32;
    let pos = 0u32;

    let logits_seq = t1._forward_impl(token, pos, false, true, &mut s1);
    let logits_par = t2._forward_impl(token, pos, false, false, &mut s2);

    assert_eq!(logits_seq.len(), logits_par.len());

    let max_diff = logits_seq
        .iter()
        .zip(logits_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("Max diff at pos 0: {}", max_diff);
    assert!(
        max_diff < 1e-3,
        "forward_sequential and forward should produce similar results at pos 0"
    );
}

#[test]
fn test_forward_compare_at_pos_one() {
    let t1 = MockTransformer::new();
    let t2 = MockTransformer::new();
    let mut s1 = InferenceSession::new(t1.base.shape.as_ref().unwrap());
    let mut s2 = InferenceSession::new(t2.base.shape.as_ref().unwrap());

    let token = 1u32;
    let pos = 1u32;

    let logits_seq = t1._forward_impl(token, pos, false, true, &mut s1);
    let logits_par = t2._forward_impl(token, pos, false, false, &mut s2);

    let max_diff = logits_seq
        .iter()
        .zip(logits_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("Max diff at pos 1: {}", max_diff);
    assert!(
        max_diff < 1e-3,
        "forward_sequential and forward should produce similar results at pos 1"
    );
}

#[test]
fn test_forward_compare_multiple_tokens() {
    for pos in 0..3 {
        let t1 = MockTransformer::new();
        let t2 = MockTransformer::new();
        let mut s1 = InferenceSession::new(t1.base.shape.as_ref().unwrap());
        let mut s2 = InferenceSession::new(t2.base.shape.as_ref().unwrap());

        let token = 1u32;

        let logits_seq = t1._forward_impl(token, pos, false, true, &mut s1);
        let logits_par = t2._forward_impl(token, pos, false, false, &mut s2);

        let max_diff = logits_seq
            .iter()
            .zip(logits_par.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!("Max diff at pos {}: {}", pos, max_diff);
        assert!(
            max_diff < 1e-3,
            "forward_sequential and forward should produce similar results at pos {}",
            pos
        );
    }
}

#[test]
fn test_forward_deterministic() {
    let transformer = MockTransformer::new();
    let mut session = InferenceSession::new(transformer.base.shape.as_ref().unwrap());
    let token = 1u32;
    let pos = 1u32;

    let logits1 = transformer._forward_impl(token, pos, false, true, &mut session);
    let logits2 = transformer._forward_impl(token, pos, false, true, &mut session);

    let max_diff = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    assert!(max_diff < 1e-6, "forward should be deterministic");
}

fn main() {
    test_forward_compare_at_pos_zero();
    test_forward_compare_at_pos_one();
    test_forward_compare_multiple_tokens();
    println!("All tests passed!");
}
