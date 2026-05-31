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


use crate::base_transformer::{InferenceSession, ModelWeights, TransformerShape};
use crate::quantization::{MutableQuantizedTensorQ8, quantize as q8_quantize};
use crate::weight_type::{
    matmul_f32, matmul_q8_slices, q4_0_quantize,
    QuantizedTensor, MutableQuantizedTensorQ4_0, WeightFormat,
};
use crate::util::rmsnorm;
use rayon::prelude::*;
use wide::f32x8;

trait ForwardKernel {
    fn quantize(dst: &mut QuantizedTensor, x: &[f32], n: usize, gs: u32);
    fn matmul(out: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize);
    fn workspace_buffer(len: usize, gs: usize) -> QuantizedTensor;
}

struct Q8Kernel;
impl ForwardKernel for Q8Kernel {
    #[inline(always)]
    fn quantize(dst: &mut QuantizedTensor, x: &[f32], n: usize, gs: u32) {
        let QuantizedTensor::Q8(ref mut q) = dst else { unsafe { std::hint::unreachable_unchecked() } };
        q8_quantize(q, x, n, gs);
    }
    #[inline(always)]
    fn matmul(out: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize) {
        let QuantizedTensor::Q8(ref xq) = x else { unsafe { std::hint::unreachable_unchecked() } };
        let QuantizedTensor::Q8(ref wq) = w else { unsafe { std::hint::unreachable_unchecked() } };
        matmul_q8_slices(out, &xq.quant_vals, &xq.scale_factor, &wq.quant_vals, &wq.scale_factor, n, o, gs);
    }
    #[inline(always)]
    fn workspace_buffer(len: usize, gs: usize) -> QuantizedTensor {
        QuantizedTensor::Q8(MutableQuantizedTensorQ8 {
            quant_vals: vec![0i8; len],
            scale_factor: vec![0.0f32; len / gs],
        })
    }
}

struct Q4_0Kernel;
impl ForwardKernel for Q4_0Kernel {
    #[inline(always)]
    fn quantize(dst: &mut QuantizedTensor, x: &[f32], n: usize, gs: u32) {
        let QuantizedTensor::Q4_0(ref mut q) = dst else { unsafe { std::hint::unreachable_unchecked() } };
        q4_0_quantize(q, x, n, gs);
    }
    #[inline(always)]
    fn matmul(out: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize) {
        let QuantizedTensor::Q4_0(ref xq) = x else { unsafe { std::hint::unreachable_unchecked() } };
        let QuantizedTensor::Q4_0(ref wq) = w else { unsafe { std::hint::unreachable_unchecked() } };
        matmul_q8_slices(out, &xq.quant_vals, &xq.scale_factor, &wq.quant_vals, &wq.scale_factor, n, o, gs);
    }
    #[inline(always)]
    fn workspace_buffer(len: usize, _gs: usize) -> QuantizedTensor {
        QuantizedTensor::Q4_0(MutableQuantizedTensorQ4_0 {
            quant_vals: vec![0i8; len],
            scale_factor: vec![0.0f32; len / _gs],
        })
    }
}

struct F16Kernel;
impl ForwardKernel for F16Kernel {
    #[inline(always)]
    fn quantize(dst: &mut QuantizedTensor, x: &[f32], n: usize, _gs: u32) {
        let QuantizedTensor::F16(ref mut dst_vec) = dst else { unsafe { std::hint::unreachable_unchecked() } };
        dst_vec[..n].copy_from_slice(&x[..n]);
    }
    #[inline(always)]
    fn matmul(out: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, _gs: usize) {
        let QuantizedTensor::F16(ref xv) = x else { unsafe { std::hint::unreachable_unchecked() } };
        let QuantizedTensor::F16(ref wv) = w else { unsafe { std::hint::unreachable_unchecked() } };
        matmul_f32(out, xv, wv, n, o);
    }
    #[inline(always)]
    fn workspace_buffer(len: usize, _gs: usize) -> QuantizedTensor {
        QuantizedTensor::F16(vec![0.0f32; len])
    }
}

struct ForwardWorkspace {
    sq: Vec<f32>,
    scores: Vec<f32>,
    att_output: Vec<f32>,
    x_q: QuantizedTensor,
    hidden_gate: Vec<f32>,
    hidden_up: Vec<f32>,
    ffn_output: Vec<f32>,
}

impl ForwardWorkspace {
    fn new<K: ForwardKernel>(batch: usize, dim: usize, hidden_dim: usize, att_dim: usize, n_heads: usize, seq_len: usize, gs: usize) -> Self {
        let max_q = (batch * dim).max(batch * att_dim).max(batch * hidden_dim);
        Self {
            sq: vec![0.0f32; batch * att_dim],
            scores: vec![0.0f32; n_heads * batch * seq_len],
            att_output: vec![0.0f32; batch * att_dim],
            x_q: K::workspace_buffer(max_q, gs),
            hidden_gate: vec![0.0f32; batch * hidden_dim],
            hidden_up: vec![0.0f32; batch * hidden_dim],
            ffn_output: vec![0.0f32; batch * dim],
        }
    }
}

#[inline(always)]
fn llama_forward_x_inner<K: ForwardKernel>(
    rope_fn: fn(&TransformerShape, &mut [f32], &mut [f32], u32),
    tokens: &[u32],
    start_pos: u32,
    session: &mut InferenceSession,
    shape: &TransformerShape,
    weights: &ModelWeights,
    get_embeddings_batch: &dyn Fn(&[u32]) -> Vec<Vec<f32>>,
) -> Vec<f32> {
    let batch = tokens.len();
    let dim = shape.dimension as usize;
    let hidden_dim = shape.hidden_dimension as usize;
    let att_dim = (shape.n_heads * shape.head_size) as usize;
    let kv_dim = (shape.n_kv_heads * shape.head_size) as usize;
    let gs = shape.group_size as usize;
    let seq_len = start_pos as usize + batch;

    let mut evolving_state_through_layers: Vec<f32> = get_embeddings_batch(tokens)
        .into_iter().flatten().collect();
    let mut x_normalized = vec![0.0f32; batch * dim];
    let mut ws = ForwardWorkspace::new::<K>(batch, dim, hidden_dim, att_dim, shape.n_heads as usize, seq_len, gs);

    let attn_scale = if shape.attention_scale != 0.0 {
        shape.attention_scale
    } else {
        1.0 / (shape.head_size as f32).sqrt()
    };

    for layer in 0..shape.n_layers as usize {
        let cache_off = layer * shape.ctx_len as usize * kv_dim;
        let write_off = start_pos as usize * kv_dim;
        let total = batch * dim;
        let n_simd = total / 8;

        // RMS norm
        x_normalized.par_chunks_mut(dim).enumerate().for_each(|(t, n)| {
            rmsnorm(n, &evolving_state_through_layers[t * dim..][..dim], &weights.w_rms_att[layer], dim, shape.rms_norm_eps, false);
        });

        // Quantize
        K::quantize(&mut ws.x_q, &x_normalized, batch * dim, gs as u32);

        // Q, K, V projections
        ws.sq[..batch * att_dim].fill(0.0);
        K::matmul(&mut ws.sq[..batch * att_dim], &ws.x_q, &weights.wq[layer], dim, att_dim, gs);

        let k = &mut session.key_cache[cache_off + write_off..][..batch * kv_dim];
        let v = &mut session.value_cache[cache_off + write_off..][..batch * kv_dim];
        K::matmul(k, &ws.x_q, &weights.wk[layer], dim, kv_dim, gs);
        K::matmul(v, &ws.x_q, &weights.wv[layer], dim, kv_dim, gs);

        // RoPE
        ws.sq.par_chunks_mut(att_dim)
            .zip(k.par_chunks_mut(kv_dim))
            .enumerate()
            .for_each(|(t, (q, kk))| rope_fn(shape, q, kk, start_pos + t as u32));

        // Attention
        let k_all = &session.key_cache[cache_off..][..seq_len * kv_dim];
        let v_all = &session.value_cache[cache_off..][..seq_len * kv_dim];
        fused_attention(
            &mut ws.att_output[..batch * att_dim],
            &mut ws.scores[..shape.n_heads as usize * batch * seq_len],
            &ws.sq[..batch * att_dim],
            k_all, v_all,
            seq_len, batch,
            shape.n_heads as usize, shape.n_kv_heads as usize,
            shape.head_size as usize, start_pos, kv_dim, attn_scale,
        );

        // WO projection
        K::quantize(&mut ws.x_q, &ws.att_output[..batch * att_dim], batch * att_dim, gs as u32);
        ws.ffn_output[..batch * dim].fill(0.0);
        K::matmul(&mut ws.ffn_output[..batch * dim], &ws.x_q, &weights.wo[layer], att_dim, dim, gs);

        // Residual add (in-place on evolving_state)
        for s in 0..n_simd {
            let b = s * 8;
            let xv = f32x8::from(&evolving_state_through_layers[b..b + 8]);
            let fv = f32x8::from(&ws.ffn_output[b..b + 8]);
            (xv + fv).to_array().iter().enumerate().for_each(|(j, v)| evolving_state_through_layers[b + j] = *v);
        }
        for i in (n_simd * 8)..total {
            evolving_state_through_layers[i] += ws.ffn_output[i];
        }

        // RMS norm (post-attention)
        x_normalized.par_chunks_mut(dim).enumerate().for_each(|(t, n)| {
            rmsnorm(n, &evolving_state_through_layers[t * dim..][..dim], &weights.w_rms_post_att[layer], dim, shape.rms_norm_eps, false);
        });

        // FFN quantize
        K::quantize(&mut ws.x_q, &x_normalized, batch * dim, gs as u32);

        // FFN gate/up
        ws.hidden_gate[..batch * hidden_dim].fill(0.0);
        ws.hidden_up[..batch * hidden_dim].fill(0.0);
        K::matmul(&mut ws.hidden_gate[..batch * hidden_dim], &ws.x_q, &weights.w1[layer], dim, hidden_dim, gs);
        K::matmul(&mut ws.hidden_up[..batch * hidden_dim], &ws.x_q, &weights.w3[layer], dim, hidden_dim, gs);

        // SiLU(gate) * up in-place on gate buffer
        ws.hidden_gate[..batch * hidden_dim]
            .par_chunks_exact_mut(hidden_dim)
            .zip(ws.hidden_up[..batch * hidden_dim].par_chunks_exact(hidden_dim))
            .for_each(|(gate, up)| {
                let n_simd = hidden_dim / 8;
                for s in 0..n_simd {
                    let b = s * 8;
                    let g = f32x8::from(&gate[b..b + 8]);
                    let u = f32x8::from(&up[b..b + 8]);
                    let silu = g / (f32x8::splat(1.0) + (-g).exp());
                    let r = (silu * u).to_array();
                    gate[b..b + 8].copy_from_slice(&r);
                }
                for i in (n_simd * 8)..hidden_dim {
                    gate[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
                }
            });

        // FFN down projection
        K::quantize(&mut ws.x_q, &ws.hidden_gate[..batch * hidden_dim], batch * hidden_dim, gs as u32);
        ws.ffn_output[..batch * dim].fill(0.0);
        K::matmul(&mut ws.ffn_output[..batch * dim], &ws.x_q, &weights.w2[layer], hidden_dim, dim, gs);

        // Residual add (in-place)
        for s in 0..n_simd {
            let b = s * 8;
            let xv = f32x8::from(&evolving_state_through_layers[b..b + 8]);
            let fv = f32x8::from(&ws.ffn_output[b..b + 8]);
            (xv + fv).to_array().iter().enumerate().for_each(|(j, v)| evolving_state_through_layers[b + j] = *v);
        }
        for i in (n_simd * 8)..total {
            evolving_state_through_layers[i] += ws.ffn_output[i];
        }
    }

    // Final norm
    x_normalized.par_chunks_mut(dim).enumerate().for_each(|(t, n)| {
        rmsnorm(n, &evolving_state_through_layers[t * dim..][..dim], &weights.w_rms_final, dim, shape.rms_norm_eps, false);
    });

    // Logits (last token, via tied embedding weight)
    let last = &x_normalized[(batch - 1) * dim..batch * dim];
    let vocab = shape.vocab_size as usize;
    let mut logits = vec![0.0f32; vocab];
    let n_simd = dim / 8;
    for v in 0..vocab {
        let emb_off = v * dim;
        let mut sum = f32x8::ZERO;
        for s in 0..n_simd {
            let b = s * 8;
            sum += f32x8::from(&last[b..b + 8]) * f32x8::from(&weights.token_embedding[emb_off + b..][..8]);
        }
        let mut acc = sum.reduce_add();
        for d in (n_simd * 8)..dim {
            acc += last[d] * weights.token_embedding[emb_off + d];
        }
        logits[v] = acc;
    }
    logits
}

pub fn llama_forward_x(
    fmt: WeightFormat,
    rope_fn: fn(&TransformerShape, &mut [f32], &mut [f32], u32),
    tokens: &[u32],
    start_pos: u32,
    session: &mut InferenceSession,
    shape: &TransformerShape,
    weights: &ModelWeights,
    get_embeddings_batch: &dyn Fn(&[u32]) -> Vec<Vec<f32>>,
) -> Vec<f32> {
    match fmt {
        WeightFormat::Q8_0 => llama_forward_x_inner::<Q8Kernel>(rope_fn, tokens, start_pos, session, shape, weights, get_embeddings_batch),
        WeightFormat::Q4_0 => llama_forward_x_inner::<Q4_0Kernel>(rope_fn, tokens, start_pos, session, shape, weights, get_embeddings_batch),
        WeightFormat::F16 => llama_forward_x_inner::<F16Kernel>(rope_fn, tokens, start_pos, session, shape, weights, get_embeddings_batch),
    }
}

fn fused_attention(
    output: &mut [f32],
    scores: &mut [f32],
    sq: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    seq_len: usize,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_size: usize,
    start_pos: u32,
    kv_dim: usize,
    attn_scale: f32,
) {
    let kv_mul = n_heads / n_kv_heads;

    scores.par_chunks_mut(seq_len).enumerate().for_each(|(idx, score_row)| {
        let qh = idx / batch;
        let t1 = idx % batch;
        let kvh = qh / kv_mul;
        let abs_pos = start_pos as usize + t1;
        let max_t2 = abs_pos.min(seq_len - 1);

        let q_base = t1 * (n_heads * head_size) + qh * head_size;
        let q = &sq[q_base..q_base + head_size];

        let mut max_score = -1e10f32;

        for t2 in 0..=max_t2 {
            let k_base = t2 * kv_dim + kvh * head_size;
            let k = &key_cache[k_base..k_base + head_size];

            let n_simd = head_size / 8;
            let mut sum_vec = f32x8::ZERO;
            for s in 0..n_simd {
                let d = s * 8;
                let q_vec = f32x8::from(&q[d..d + 8]);
                let k_vec = f32x8::from(&k[d..d + 8]);
                sum_vec += q_vec * k_vec;
            }
            let mut sum = sum_vec.reduce_add();
            for d in (n_simd * 8)..head_size {
                sum += q[d] * k[d];
            }

            let s_val = sum * attn_scale;
            score_row[t2] = s_val;
            if s_val > max_score {
                max_score = s_val;
            }
        }

        for t2 in (max_t2 + 1)..seq_len {
            score_row[t2] = -1e10;
        }

        let mut sum_exp = 0.0f32;
        for t2 in 0..seq_len {
            let exp_s = (score_row[t2] - max_score).exp();
            score_row[t2] = exp_s;
            sum_exp += exp_s;
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        for t2 in 0..seq_len {
            score_row[t2] *= inv_sum;
        }
    });

    output.par_chunks_mut(n_heads * head_size).enumerate().for_each(|(t1, out_token)| {
        for qh in 0..n_heads {
            let kvh = qh / kv_mul;
            let out_base = qh * head_size;
            let s_base = qh * (batch * seq_len) + t1 * seq_len;

            for d in 0..head_size {
                let n_simd = seq_len / 8;
                let mut sum_vec = f32x8::ZERO;
                for t2_chunk in 0..n_simd {
                    let t2 = t2_chunk * 8;
                    let s_vec = f32x8::from(&scores[s_base + t2..s_base + t2 + 8]);
                    let v0 = value_cache[t2 * kv_dim + kvh * head_size + d];
                    let v1 = value_cache[(t2 + 1) * kv_dim + kvh * head_size + d];
                    let v2 = value_cache[(t2 + 2) * kv_dim + kvh * head_size + d];
                    let v3 = value_cache[(t2 + 3) * kv_dim + kvh * head_size + d];
                    let v4 = value_cache[(t2 + 4) * kv_dim + kvh * head_size + d];
                    let v5 = value_cache[(t2 + 5) * kv_dim + kvh * head_size + d];
                    let v6 = value_cache[(t2 + 6) * kv_dim + kvh * head_size + d];
                    let v7 = value_cache[(t2 + 7) * kv_dim + kvh * head_size + d];
                    let v_vec = f32x8::new([v0, v1, v2, v3, v4, v5, v6, v7]);
                    sum_vec += s_vec * v_vec;
                }
                let mut sum = sum_vec.reduce_add();
                for t2 in (n_simd * 8)..seq_len {
                    sum += scores[s_base + t2]
                        * value_cache[t2 * kv_dim + kvh * head_size + d];
                }
                out_token[out_base + d] = sum;
            }
        }
    });
}
