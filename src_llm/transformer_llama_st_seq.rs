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


// Educational reference implementation of the LLaMA forward pass.
// This is a sequential (single-threaded), scalar version with extensive inline comments
// to help understand the transformer architecture. For actual use, see the parallel
// forward pass in forward.rs or the ST (Safetensors) loader in transformer_llama_st.rs.
//
// This variant loads weights from Safetensors format (HuggingFace) instead of lm.rs.

use crate::base_transformer::{
    InferenceSession, ModelFamily, ModelWeights, TransformerBase, TransformerShape,
};
use crate::quantization::{quantize, MutableQuantizedTensorQ8, QuantType};
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use crate::util::rmsnorm;
use crate::util::silu;
use crate::weight_type::{QuantizedTensor, WeightFormat};
#[cfg(feature = "debug_prints")]
use crate::util::{print_byte_mem, print_float_matrix_3d, PrintFormat, print_float_mem, PrintRange};
use memmap2::Mmap;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::path::Path;

const LLAMA_BOS: u32 = 1;
const LLAMA_EOS: u32 = 2;

pub struct TransformerLlamaStSeq {
    base: TransformerBase,
    #[allow(dead_code)]
    weight_format: WeightFormat,
}

impl TransformerLlamaStSeq {
    pub fn load_model_with_override(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, fmt_override)
    }

    fn load_model_inner(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = Path::new(model_path);
        if !model_path.exists() {
            return Err(format!("Model file does not exist: {}", model_path.display()).into());
        }
        let model_dir = model_path.parent().unwrap();

        // --- Step 1: Read config.json ---
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_text)?;

        let dim = config["hidden_size"].as_u64().unwrap() as u32;
        let hidden_dim = config["intermediate_size"].as_u64().unwrap() as u32;
        let n_heads = config["num_attention_heads"].as_u64().unwrap() as u32;
        let n_kv_heads = config["num_key_value_heads"].as_u64().unwrap() as u32;
        let head_size = config["head_dim"].as_u64().unwrap() as u32;
        let n_layers = config["num_hidden_layers"].as_u64().unwrap() as u32;
        let vocab_size = config["vocab_size"].as_u64().unwrap() as u32;
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap() as f32;
        let rope_theta = config["rope_theta"].as_f64().unwrap() as f32;
        let max_seq_len = config["max_position_embeddings"].as_u64().unwrap() as u32;

        // Only Q8_0 is supported by the sequential forward pass
        let (q_type, weight_format, group_size) = match fmt_override {
            Some(f) => match f {
                WeightFormat::Q8_0 => (QuantType::Q8_0, WeightFormat::Q8_0,
                    config.get("quantization").and_then(|q| q.get("group_size")).and_then(|v| v.as_u64()).unwrap_or(32) as u32),
                _ => {
                    eprintln!("Warning: ST sequential only supports Q8_0, ignoring override to {:?}", f);
                    (QuantType::Q8_0, WeightFormat::Q8_0, 32)
                }
            },
            None => {
                let q = config.get("quantization");
                match q {
                    None => {
                        eprintln!("Warning: ST sequential only supports Q8_0 quantized models, found F16");
                        (QuantType::Q8_0, WeightFormat::Q8_0, 32)
                    }
                    Some(q_obj) => (QuantType::Q8_0, WeightFormat::Q8_0,
                        q_obj.get("group_size").and_then(|v| v.as_u64()).unwrap_or(32) as u32),
                }
            }
        };

        println!("Loading Llama Safetensors model (sequential) from: {}", model_path.display());
        println!("Config: dim={}, layers={}, heads={}, kv_heads={}, fmt={:?}",
                 dim, n_layers, n_heads, n_kv_heads, weight_format);

        let ctx_len = max_seq_len.min(max_ctx_len);

        let shape = TransformerShape {
            dimension: dim,
            hidden_dimension: hidden_dim,
            n_heads,
            head_size,
            n_kv_heads,
            vocab_size,
            n_layers,
            ctx_len,
            rms_norm_eps,
            rope_theta,
            q_type,
            group_size,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            rope_original_ctx_len: ctx_len,
            attention_scale: 0.0,
        };

        // --- Step 2: Open safetensors file ---
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Safetensors format: 8 bytes header_len (u64 LE) | JSON header | raw data
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&mmap[8..8 + header_len])?;
        let header: serde_json::Value = serde_json::from_str(header_str)?;

        // Build a lookup table: tensor name → (dtype, shape, byte offset in mmap)
        struct TensorInfo {
            dtype: String,
            shape: Vec<usize>,
            offset: usize,
        }

        let data_start = 8 + header_len;
        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();

        if let Some(obj) = header.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" { continue; }
                let dtype = info["dtype"].as_str().unwrap().to_string();
                let shape: Vec<usize> = info["shape"].as_array().unwrap()
                    .iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let offsets = info["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap() as usize;
                tensors.insert(name.clone(), TensorInfo {
                    dtype, shape, offset: data_start + start,
                });
            }
        }

        // --- Step 3: F16 → F32 helper for norm weights ---
        fn load_f16_vec(data: &[u8], start: usize, len: usize) -> Vec<f32> {
            let mut result = vec![0.0f32; len];
            for i in 0..len {
                let bits = u16::from_le_bytes(data[start + i * 2..][..2].try_into().unwrap());
                result[i] = half::f16::from_bits(bits).to_f32();
            }
            result
        }

        let dim_usize = dim as usize;
        let n_layers_usize = n_layers as usize;

        // --- Step 4: Load final RMS norm ---
        let w_rms_final = {
            let ti = &tensors["model.norm.weight"];
            assert_eq!(ti.dtype, "F16");
            assert_eq!(ti.shape[0], dim_usize);
            load_f16_vec(&mmap, ti.offset, dim_usize)
        };

        // --- Step 5: Load per-layer attention norms ---
        let mut w_rms_att = Vec::with_capacity(n_layers_usize);
        let mut w_rms_post_att = Vec::with_capacity(n_layers_usize);
        for i in 0..n_layers_usize {
            let att = {
                let ti = &tensors[&format!("model.layers.{}.input_layernorm.weight", i)];
                load_f16_vec(&mmap, ti.offset, dim_usize)
            };
            let post = {
                let ti = &tensors[&format!("model.layers.{}.post_attention_layernorm.weight", i)];
                load_f16_vec(&mmap, ti.offset, dim_usize)
            };
            w_rms_att.push(att);
            w_rms_post_att.push(post);
        }

        // --- Step 6: Load token embedding ---
        let (token_embedding, token_embedding_quant) = {
            let wt = &tensors["model.embed_tokens.weight"];
            let st = &tensors["model.embed_tokens.scales"];
            let bt = &tensors["model.embed_tokens.biases"];
            let embed_qt = crate::util::load_q8_tensor_st(
                &mmap, vocab_size as usize, dim_usize, group_size as usize,
                wt.offset, st.offset, bt.offset,
            );
            let gs = group_size as usize;
            let qv = &embed_qt.quant_vals;
            let sf = &embed_qt.scale_factor;
            let mut emb = vec![0.0; vocab_size as usize * dim_usize];
            for g in 0..(emb.len() / gs) {
                let scale = sf[g];
                let base = g * gs;
                for j in 0..gs {
                    emb[base + j] = qv[base + j] as f32 * scale;
                }
            }
            (emb, Some(QuantizedTensor::Q8(embed_qt)))
        };

        // --- Step 7: Load per-layer Q/K/V/O and FFN weights ---
        let gs = group_size as usize;
        let q_rows = n_heads as usize * head_size as usize;
        let kv_rows = n_kv_heads as usize * head_size as usize;

        let mut wq = Vec::with_capacity(n_layers_usize);
        let mut wk = Vec::with_capacity(n_layers_usize);
        let mut wv = Vec::with_capacity(n_layers_usize);
        let mut wo = Vec::with_capacity(n_layers_usize);
        let mut w1 = Vec::with_capacity(n_layers_usize);
        let mut w2 = Vec::with_capacity(n_layers_usize);
        let mut w3 = Vec::with_capacity(n_layers_usize);

        for i in 0..n_layers_usize {
            let prefix = format!("model.layers.{}", i);

            let load_q = |proj: &str, rows: usize, cols: usize| -> QuantizedTensor {
                let base = format!("{}.{}", prefix, proj);
                let w = &tensors[&format!("{}.weight", base)];
                let s = &tensors[&format!("{}.scales", base)];
                let b = &tensors[&format!("{}.biases", base)];
                QuantizedTensor::Q8(crate::util::load_q8_tensor_st(
                    &mmap, rows, cols, gs, w.offset, s.offset, b.offset,
                ))
            };

            wq.push(load_q("self_attn.q_proj", q_rows, dim_usize));
            wk.push(load_q("self_attn.k_proj", kv_rows, dim_usize));
            wv.push(load_q("self_attn.v_proj", kv_rows, dim_usize));
            wo.push(load_q("self_attn.o_proj", dim_usize, q_rows));
            w1.push(load_q("mlp.gate_proj", hidden_dim as usize, dim_usize));
            w2.push(load_q("mlp.down_proj", dim_usize, hidden_dim as usize));
            w3.push(load_q("mlp.up_proj", hidden_dim as usize, dim_usize));
        }

        let model_weights = ModelWeights {
            token_embedding,
            token_embedding_quant,
            w_rms_att,
            wq, wk, wv, wo,
            w_rms_post_att,
            w1, w2, w3,
            w_rms_final,
        };

        // --- Step 8: Load tokenizer ---
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_json(&tokenizer_path)?
        } else {
            Tokenizer::new(&model_dir.join("tokenizer.bin"))?
        };

        println!("Model loaded successfully.");

        Ok(Self {
            weight_format,
            base: TransformerBase {
                model_info: format!("Llama ST model (sequential) loaded from {}", model_path.display()),
                model_family: Some(ModelFamily::Llama),
                shape: Some(shape),
                weights: Some(model_weights),
                tokenizer: Some(tokenizer),
            },
        })
    }
}

impl Transformer for TransformerLlamaStSeq {
    fn load_model(model_path: &str, max_ctx_len: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, None)
    }

    fn name(&self) -> &str {
        "TransformerLlamaStSeq"
    }

    fn vocab_size(&self) -> usize {
        self.base
            .shape
            .as_ref()
            .map(|s| s.vocab_size as usize)
            .unwrap_or(0)
    }

    fn forward_x(&self, tokens: &[u32], start_pos: u32, session: &mut InferenceSession) -> Vec<f32> {
        // LLaMA Pre-norm Data Flow (per layer):
        // ┌─────────────────────────────────────────────────────────────────┐
        // │ 1. evolving_state_through_layers (original input from prev)     │
        // │                         ↓                                       │
        // │ 2. Clone → pre_attn_state (saved for residual)                  │
        // │                         ↓                                       │
        // │ 3. RMS norm → x_normalized (used for Q,K,V projections)         │
        // │                         ↓                                       │
        // │ 4. Attention computation + WO                                   │
        // │                         ↓                                       │
        // │ 5. pre_attn_state += wo_output (CORRECT: residual to original)  │
        // │                         ↓                                       │
        // │ 6. Clone → pre_ffn_state (saved for FFN residual)               │
        // │                         ↓                                       │
        // │ 7. RMS norm (w_rms_post_att) → x_normalized (for FFN)           │
        // │                         ↓                                       │
        // │ 8. FFN computation                                              │
        // │                         ↓                                       │
        // │ 9. pre_ffn_state += ffn_output (CORRECT: residual to original)  │
        // │                         ↓                                       │
        // │ 10. evolving_state_through_layers = pre_ffn_state (layer out)   │
        // └─────────────────────────────────────────────────────────────────┘

        let _shape = self.base.shape.as_ref().expect("Shape not initialized");
        let weights = self.base.weights.as_ref().expect("Weights not initialized");
        let embeddings = self.get_embeddings_batch(tokens);
        let mut evolving_state_through_layers =
            embeddings.into_iter().flatten().collect::<Vec<f32>>();
        let mut x_normalized = vec![0.0; (_shape.dimension as usize) * tokens.len()];
        #[cfg(feature = "debug_prints")]
        println!(
            "Prefill called with tokens: {:?}, resulting in embeddings of length {}",
            tokens,
            evolving_state_through_layers.len()
        );
        #[cfg(feature = "debug_prints")]
        println!(
            "This is a matrix of shape ({} tokens, {} dimensions)",
            tokens.len(),
            evolving_state_through_layers.len() / tokens.len()
        );

        for layer in 0.._shape.n_layers as usize {
            // Save pre-normalization state for attention residual (pre-norm)
            let mut pre_attn_state = vec![0.0; _shape.dimension as usize * tokens.len()];
            pre_attn_state.copy_from_slice(&evolving_state_through_layers);

            // RMS normalize all tokens in the prompt, store in x_normalized
            for t in 0..tokens.len() {
                #[cfg(feature = "debug_prints")]
                if t == 0 {
                    print_float_mem(
                        &format!("token {} layer {} embeddings:", t, layer),
                        &evolving_state_through_layers[t * (_shape.dimension as usize)
                            ..(t + 1) * (_shape.dimension as usize)],
                        Some((PrintRange::HeadNTail, 8)),
                    );
                }

                rmsnorm(
                    &mut x_normalized
                        [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                    &evolving_state_through_layers
                        [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                    &weights.w_rms_att[layer],
                    _shape.dimension as usize,
                    _shape.rms_norm_eps,
                    false,
                );

                #[cfg(feature = "debug_prints")]
                if t == 0 {
                    print_float_mem(
                        &format!("token {} layer {} embeddings after RMS_Norm", t, layer),
                        &x_normalized[t * (_shape.dimension as usize)
                            ..(t + 1) * (_shape.dimension as usize)],
                        Some((PrintRange::HeadNTail, 8)),
                    );
                }
            }
            //We know that attention_dim = dim (model dimension) = n_heads * head_size, and kv_dim = n_kv_heads * head_size for this llama model
            // but we still treat them independently in case we want to support other models in the future where this might not be the case, and also to make the code more readable by explicitly naming these dimensions according to their role in attention
            let attention_dim = (_shape.n_heads * _shape.head_size) as usize;
            let kv_dim = (_shape.n_kv_heads * _shape.head_size) as usize;

            //KV cache is one block for all layers, so we need to calculate the offset for the current layer and current position in the cache
            let layer_cache_offset = layer * (_shape.ctx_len as usize) * kv_dim;
            let _seq_len = (start_pos as usize) + tokens.len();
            let write_offset = (start_pos as usize) * kv_dim;

            let mut sq = vec![0.0f32; attention_dim * tokens.len()];
            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "sq matrix",
                &sq,
                tokens.len(),
                _shape.n_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            let k = &mut session.key_cache[layer_cache_offset + write_offset
                ..layer_cache_offset + write_offset + tokens.len() * kv_dim];
            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "k matrix pointing to cache",
                &k,
                tokens.len(),
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            let v = &mut session.value_cache[layer_cache_offset + write_offset
                ..layer_cache_offset + write_offset + tokens.len() * kv_dim];
            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "v matrix pointing to cache",
                &v,
                tokens.len(),
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            //Q,K,V projections for all tokens in the prompt, we can do this in place on x and store the quantized versions in the workspace for later use in attention computations
            let sxq = &mut MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (_shape.dimension as usize * tokens.len()) as usize],
                scale_factor: vec![
                    0.0;
                    ((_shape.dimension as usize * tokens.len()) as usize
                        / _shape.group_size as usize) as usize
                ],
            };

            quantize(
                sxq,
                &x_normalized,
                _shape.dimension as usize * tokens.len(),
                _shape.group_size,
            );

            #[cfg(feature = "debug_prints")]
            print_byte_mem(
                "quantized normalized embeddings in X Q8:",
                &sxq.quant_vals.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                Some((PrintRange::HeadNTail, 8)),
            );

            #[cfg(feature = "debug_prints")]
            print_float_mem(
                "quantized normalized embeddings in X Scale:",
                &sxq.scale_factor
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<Vec<f32>>(),
                Some((PrintRange::HeadNTail, 8)),
            );

            //Project Q => shape n x attention_dim. we think of it in re-shape n x n_heads x head_size stored in row major.
            multiply_gqa_2d_col_major_quantized(
                &mut sq,
                sxq,
                &weights.wq[layer as usize],
                tokens.len(),
                _shape.dimension as usize,
                attention_dim,
                _shape.group_size as usize,
            );
            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "sq matrix that is projected Q from quantized matmul with Wq. We think of it in re-shape n x n_heads x head_size",
                &sq,
                tokens.len(),
                _shape.n_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            //Project K
            multiply_gqa_2d_col_major_quantized(
                k,
                sxq,
                &weights.wk[layer as usize],
                tokens.len(),
                _shape.dimension as usize,
                kv_dim,
                _shape.group_size as usize,
            );

            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "k matrix after WK projection i.e. X quantized matmul with Wk, stored in cache",
                k,
                tokens.len(),
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            //Project V
            multiply_gqa_2d_col_major_quantized(
                v,
                sxq,
                &weights.wv[layer as usize],
                tokens.len(),
                _shape.dimension as usize,
                kv_dim,
                _shape.group_size as usize,
            );

            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "v matrix after WV projection i.e. X quantized matmul with Wv, stored in cache",
                v,
                tokens.len(),
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );
            //Apply RoPE with Llama's YARN-style frequency scaling to the projected Q and K matrices, we can do this in place on the cache for K and on the sq matrix for Q
            for t in 0..tokens.len() {
                let pos_k = &mut k[t * kv_dim..(t + 1) * kv_dim];
                let pos_sq = &mut sq[t * attention_dim..(t + 1) * attention_dim];

                #[cfg(feature = "debug_prints")]
                print_float_matrix_3d(
                    &format!("sq matrix Before RoPE and pos {}", t),
                    pos_sq,
                    1,
                    _shape.n_heads as usize,
                    _shape.head_size as usize,
                    Some(PrintFormat::Summary1),
                );
                #[cfg(feature = "debug_prints")]
                print_float_matrix_3d(
                    &format!("k matrix Before RoPE and pos {}", t),
                    pos_k,
                    1,
                    _shape.n_kv_heads as usize,
                    _shape.head_size as usize,
                    Some(PrintFormat::Summary1),
                );

                rope_scalar(
                    _shape,
                    pos_sq,
                    pos_k,
                    (start_pos + t as u32).try_into().unwrap(),
                );

                #[cfg(feature = "debug_prints")]
                print_float_matrix_3d(
                    &format!("sq matrix after RoPE and pos {}", t),
                    pos_sq,
                    1,
                    _shape.n_heads as usize,
                    _shape.head_size as usize,
                    Some(PrintFormat::Summary1),
                );
                #[cfg(feature = "debug_prints")]
                print_float_matrix_3d(
                    &format!("k matrix after RoPE and pos {}", t),
                    pos_k,
                    1,
                    _shape.n_kv_heads as usize,
                    _shape.head_size as usize,
                    Some(PrintFormat::Summary1),
                );
            }

            let seq_len = (start_pos as usize) + tokens.len();

            let k_all = &session.key_cache
                [layer_cache_offset..layer_cache_offset + seq_len * kv_dim];
            let v_all = &session.value_cache
                [layer_cache_offset..layer_cache_offset + seq_len * kv_dim];

            let mut kt = vec![0.0f32; kv_dim * seq_len];
            mat_transpose_3d_row_major(
                &mut kt,
                k_all,
                seq_len,
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
            );

            let mut s = vec![0.0f32; _shape.n_heads as usize * tokens.len() * seq_len];

            let attention_scale = 1.0 / (_shape.head_size as f32).sqrt();
            broadcast_attention_q_kt_softmax_scalar(
                &mut s,
                &sq,
                &kt,
                tokens.len(),
                seq_len,
                _shape.n_heads as usize,
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
                attention_scale,
                start_pos,
            );

            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "S matrix after attention with causal mask",
                &s,
                _shape.n_heads as usize,
                tokens.len(),
                seq_len,
                Some(PrintFormat::Summary1),
            );

            let mut vt = vec![0.0f32; kv_dim * seq_len];
            mat_transpose_3d_row_major(
                &mut vt,
                v_all,
                seq_len,
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
            );

            let mut att_output =
                vec![0.0f32; tokens.len() * _shape.n_heads as usize * _shape.head_size as usize];
            broadcast_matmul_s_v_scalar(
                &mut att_output,
                &s,
                &vt,
                tokens.len(),
                seq_len,
                _shape.n_heads as usize,
                _shape.n_kv_heads as usize,
                _shape.head_size as usize,
            );

            #[cfg(feature = "debug_prints")]
            print_float_matrix_3d(
                "F (attention) matrix after S . V",
                &att_output,
                tokens.len(),
                _shape.n_heads as usize,
                _shape.head_size as usize,
                Some(PrintFormat::Summary1),
            );

            // Quantize attention output before WO projection
            let att_output_dim = tokens.len() * (_shape.n_heads * _shape.head_size) as usize;
            let mut quant_att_output = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; att_output_dim],
                scale_factor: vec![0.0; att_output_dim / _shape.group_size as usize],
            };
            quantize(
                &mut quant_att_output,
                &att_output,
                att_output_dim,
                _shape.group_size,
            );

            // Project WO: att_output (tokens × attention_dim) -> att_output (tokens × dim)
            let mut wo_output = vec![0.0f32; tokens.len() * _shape.dimension as usize];
            dequantize_mat_mul(
                &mut wo_output,
                &quant_att_output,
                &weights.wo[layer],
                _shape.group_size as usize,
                tokens.len(),
            );

            // Residual connection: add WO output to pre_attn_state (pre-norm pattern)
            for t in 0..tokens.len() {
                let start = t * _shape.dimension as usize;
                for i in 0..(_shape.dimension as usize) {
                    pre_attn_state[start + i] += wo_output[start + i];
                }
            }
            // Update state for FFN phase (replaces evolving_state_through_layers)
            evolving_state_through_layers.copy_from_slice(&pre_attn_state);

            // RMS Norm before FFN
            for t in 0..tokens.len() {
                rmsnorm(
                    &mut x_normalized
                        [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                    &evolving_state_through_layers
                        [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                    &weights.w_rms_post_att[layer],
                    _shape.dimension as usize,
                    _shape.rms_norm_eps,
                    false,
                );
            }

            // Save pre-FFN state for residual (pre-norm pattern)
            let mut pre_ffn_state = vec![0.0; _shape.dimension as usize * tokens.len()];
            pre_ffn_state.copy_from_slice(&evolving_state_through_layers);

            // FFN: Feed-Forward Network with SwiGLU activation
            //
            // ARCHITECTURE OVERVIEW:
            // The Llama FFN uses a gated linear unit (SwiGLU) which provides better expressivity
            // than traditional ReLU/GELU activations. The structure is:
            //
            //   hidden = SiLU(w1 @ x) * (w3 @ x)  // SwiGLU activation
            //   output = w2 @ hidden               // Down-projection back to dim
            //
            // COMPONENT BREAKDOWN:
            // - w1 (gate projection): dim -> hidden_dim, followed by SiLU activation
            //   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            //   This acts as a learnable gate that controls information flow
            //
            // - w3 (up projection): dim -> hidden_dim, linear (no activation)
            //   Projects input to high-dimensional space for feature extraction
            //
            // - w2 (down projection): hidden_dim -> dim
            //   Projects back to original dimension for residual connection
            //
            // WHY SWIGLU?
            // - Provides adaptive gating (learns what information to pass/block)
            // - Better gradient flow than ReLU (no dying neurons)
            // - Element-wise multiplication of gate and up-projection creates
            //   rich non-linear interactions between features
            //
            // MATRIX DIMENSIONS:
            // - Input: x_normalized (tokens × dim)
            // - w1, w3 weights: (dim × hidden_dim) in column-major
            // - w2 weights: (hidden_dim × dim) in column-major
            // - Output: ffn_output (tokens × dim)

            let hidden_dim = _shape.hidden_dimension as usize;
            let dim = _shape.dimension as usize;
            let batch = tokens.len();

            // STEP 1: Gate projection - w1 @ x -> hidden_gate (tokens × hidden_dim)
            // Quantize input for w1 projection
            let mut x_norm_q = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; batch * dim],
                scale_factor: vec![0.0; (batch * dim) / _shape.group_size as usize],
            };
            quantize(&mut x_norm_q, &x_normalized, batch * dim, _shape.group_size);

            // w1 projects from dim -> hidden_dim, output shape: (tokens × hidden_dim)
            let mut hidden_gate = vec![0.0f32; batch * hidden_dim];
            dequantize_mat_mul(
                &mut hidden_gate,
                &x_norm_q,
                &weights.w1[layer],
                _shape.group_size as usize,
                batch,
            );

            // STEP 2: Up projection - w3 @ x -> hidden_up (tokens × hidden_dim)
            // Reuse the same quantized input since w3 has the same input shape as w1
            let mut hidden_up = vec![0.0f32; batch * hidden_dim];
            dequantize_mat_mul(
                &mut hidden_up,
                &x_norm_q,
                &weights.w3[layer],
                _shape.group_size as usize,
                batch,
            );

            // STEP 3: Apply SiLU activation to gate - hidden_gate = SiLU(hidden_gate)
            // SiLU(x) = x / (1 + exp(-x)) provides smooth, non-saturating gating
            // Unlike ReLU, it preserves negative information with small weights
            for t in 0..batch {
                let start = t * hidden_dim;
                silu(&mut hidden_gate[start..start + hidden_dim]);
            }

            // STEP 4: Element-wise multiplication - hidden_gate *= hidden_up
            // This combines the gated signal with the up-projection:
            // Each element in hidden_gate acts as a learnable weight for the
            // corresponding element in hidden_up, creating adaptive feature selection
            for i in 0..(batch * hidden_dim) {
                hidden_gate[i] *= hidden_up[i];
            }

            // STEP 5: Down projection - w2 @ hidden_gate -> ffn_output (tokens × dim)
            // Quantize the SwiGLU output for w2 projection
            let mut hidden_gate_q = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; batch * hidden_dim],
                scale_factor: vec![0.0; (batch * hidden_dim) / _shape.group_size as usize],
            };
            quantize(
                &mut hidden_gate_q,
                &hidden_gate,
                batch * hidden_dim,
                _shape.group_size,
            );

            // w2 projects from hidden_dim -> dim, output shape: (tokens × dim)
            let mut ffn_output = vec![0.0f32; batch * dim];
            dequantize_mat_mul(
                &mut ffn_output,
                &hidden_gate_q,
                &weights.w2[layer],
                _shape.group_size as usize,
                batch,
            );

            // STEP 6: Residual connection - pre_ffn_state += ffn_output (pre-norm pattern)
            // Add FFN output to the pre-FFN state (not the normalized state)
            for i in 0..(batch * dim) {
                pre_ffn_state[i] += ffn_output[i];
            }
            // Update state for next layer
            evolving_state_through_layers.copy_from_slice(&pre_ffn_state);
        }

        // Final RMS Norm after all layers
        // Normalizes the final hidden states before projecting to logits
        // Uses shared weights (w_rms_final) applied to all tokens
        for t in 0..tokens.len() {
            rmsnorm(
                &mut x_normalized
                    [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                &evolving_state_through_layers
                    [t * (_shape.dimension as usize)..(t + 1) * (_shape.dimension as usize)],
                &weights.w_rms_final,
                _shape.dimension as usize,
                _shape.rms_norm_eps,
                false,
            );
        }

        // Compute logits for the LAST token only
        // In inference, we only need to predict the next token, so we only
        // compute logits from the final position in the sequence.
        //
        // Uses token_embedding transpose (weight tying):
        //   logits = embedding @ last_token_normed
        // token_embedding shape: vocab × dim
        // For each vocab index v: logits[v] = dot(last_token_normed, token_embedding[v])
        let last_token_idx = tokens.len() - 1;
        let last_token_normed = &x_normalized[last_token_idx * (_shape.dimension as usize)
            ..(last_token_idx + 1) * (_shape.dimension as usize)];

        let vocab_size = _shape.vocab_size as usize;
        let dim = _shape.dimension as usize;

        let mut logits = vec![0.0; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let embed_offset = v * dim;
            for d in 0..dim {
                sum += last_token_normed[d] * weights.token_embedding[embed_offset + d];
            }
            logits[v] = sum;
        }
        logits
    }

    fn create_session(&self) -> InferenceSession {
        InferenceSession::new(self.base.shape.as_ref().expect("Shape not initialized"))
    }

    fn get_embeddings(&self, token: u32) -> Vec<f32> {
        self.base.get_embeddings(token)
    }

    fn get_embeddings_batch(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.base.get_embeddings_batch(tokens)
    }

    fn max_ctx_len(&self) -> u32 {
        self.base.shape.as_ref().map(|s| s.ctx_len).unwrap_or(2048)
    }

    fn eos_token(&self) -> u32 {
        self.tokenizer().map(|t| t.eos).unwrap_or(LLAMA_EOS)
    }

    fn bos_token(&self) -> u32 {
        self.tokenizer().map(|t| t.bos).unwrap_or(LLAMA_BOS)
    }

    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.base.tokenizer.as_ref()
    }
}

impl TransformerLlamaStSeq {
    pub fn to_string(&self) -> String {
        self.base.to_string()
    }
}

pub fn multiply_gqa_2d_col_major_quantized(
    output: &mut [f32],
    input: &MutableQuantizedTensorQ8,
    weights: &QuantizedTensor, //stored in column major
    in_rows: usize,
    dim_in: usize,     //equal in cols also to weight rows & to model dim
    dim_out: usize,    //equal to weight cols also to output cols1
    group_size: usize, // block size in GQA
) {
    let w = match weights {
        QuantizedTensor::Q8(ref q) => q,
        _ => panic!("LMRS sequential: only Q8_0 format is currently supported"),
    };
    assert_eq!(input.quant_vals.len(), in_rows * dim_in);
    assert_eq!(output.len(), in_rows * dim_out);
    assert_eq!(w.quant_vals.len(), dim_in * dim_out);

    for row in 0..in_rows {
        let in_offset = row * dim_in;
        let out_offset = row * dim_out;

        for out_cell in 0..dim_out {
            let w_col_start = out_cell * dim_in;
            let mut dot_product_q = 0.0f32;
            for in_col in 0..dim_in {
                let x_val = input.quant_vals[in_offset + in_col] as i32;
                let w_val = w.quant_vals[w_col_start + in_col] as i32;
                let x_scale = input.scale_factor[(in_offset + in_col) / group_size];
                let w_scale = w.scale_factor[(w_col_start + in_col) / group_size];
                dot_product_q += x_val as f32 * w_val as f32 * x_scale * w_scale;
            }
            output[out_offset + out_cell] = dot_product_q;
        }
    }
}

/// Broadcast matmul: S [q_heads, q_tokens, kv_tokens] · VT [kv_heads, head_size, kv_tokens]
///
/// Computes output = S @ V where V is pre-transposed.
/// Each q_head uses the corresponding kv_head (q_head // repeat_factor) without repeating V in memory.
/// Returns output [q_tokens, q_heads, head_size]
pub fn broadcast_matmul_s_v_scalar(
    output: &mut [f32], // [q_tokens, q_heads, head_size]
    s: &[f32],          // [q_heads, q_tokens, kv_tokens]
    vt: &[f32],         // [kv_heads, head_size, kv_tokens]
    q_tokens: usize,
    kv_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_size: usize,
) {
    let repeat_factor = q_heads / kv_heads;
    assert_eq!(q_heads, kv_heads * repeat_factor);
    assert_eq!(output.len(), q_tokens * q_heads * head_size);
    assert_eq!(s.len(), q_heads * q_tokens * kv_tokens);
    assert_eq!(vt.len(), kv_heads * head_size * kv_tokens);

    let s_head_stride = q_tokens * kv_tokens;
    let s_token_stride = kv_tokens;

    let vt_head_stride = head_size * kv_tokens;
    let vt_dim_stride = kv_tokens;

    let out_token_stride = q_heads * head_size;
    let out_head_stride = head_size;

    for t1 in 0..q_tokens {
        for qh in 0..q_heads {
            let kvh = qh / repeat_factor;
            let s_base = qh * s_head_stride + t1 * s_token_stride;
            let vt_base = kvh * vt_head_stride;
            let out_base = t1 * out_token_stride + qh * out_head_stride;

            for d in 0..head_size {
                let mut sum = 0.0f32;
                for t2 in 0..kv_tokens {
                    let s_val = s[s_base + t2];
                    let v_val = vt[vt_base + d * vt_dim_stride + t2];
                    sum += s_val * v_val;
                }
                output[out_base + d] = sum;
            }
        }
    }
}

/// Fused Q·K^T → softmax per row. Computes attention scores and softmaxes
/// each (token, head) row in-place without writing intermediate results to memory.
///
/// Q shape: [q_tokens, q_heads, head_dim]
/// K^T shape: [kv_heads, head_dim, kv_tokens]
/// Scores shape: [q_heads, q_tokens, kv_tokens]
///
/// `start_pos` is the absolute position where Q tokens start in the sequence.
/// For Q token at batch index t1 (absolute position = start_pos + t1),
/// it can attend to KV tokens at positions 0..=start_pos+t1 (causal mask).
pub fn broadcast_attention_q_kt_softmax_scalar(
    scores: &mut [f32],
    q: &[f32],
    kt: &[f32],
    q_tokens: usize,
    kv_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    start_pos: u32,
) {
    let repeat_factor = q_heads / kv_heads;
    assert_eq!(q_heads, kv_heads * repeat_factor);
    assert_eq!(scores.len(), q_heads * q_tokens * kv_tokens);
    assert_eq!(q.len(), q_tokens * q_heads * head_dim);
    assert_eq!(kt.len(), kv_heads * head_dim * kv_tokens);

    let q_token_stride = q_heads * head_dim;
    let q_head_stride = head_dim;
    let kt_head_stride = head_dim * kv_tokens;
    let kt_dim_stride = kv_tokens;
    let score_head_stride = q_tokens * kv_tokens;
    let score_token_stride = kv_tokens;

    const CAUSAL_MASK_VALUE: f32 = -1e10;

    for qh in 0..q_heads {
        let kvh = qh / repeat_factor;
        let q_head_base = qh * q_head_stride;
        let kt_base = kvh * kt_head_stride;
        let score_base = qh * score_head_stride;

        for t1 in 0..q_tokens {
            let q_base = t1 * q_token_stride + q_head_base;
            let abs_pos = (start_pos as usize) + t1;

            let mut max_score = CAUSAL_MASK_VALUE;
            let max_t2 = abs_pos.min(kv_tokens - 1);

            for t2 in 0..=max_t2 {
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[q_base + d];
                    let k_val = kt[kt_base + d * kt_dim_stride + t2];
                    sum += q_val * k_val;
                }
                let s = sum * scale;
                scores[score_base + t1 * score_token_stride + t2] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            for t2 in (max_t2 + 1)..kv_tokens {
                scores[score_base + t1 * score_token_stride + t2] = CAUSAL_MASK_VALUE;
            }

            let mut sum_exp = 0.0f32;
            for t2 in 0..kv_tokens {
                let s = scores[score_base + t1 * score_token_stride + t2];
                let exp_s = (s - max_score).exp();
                scores[score_base + t1 * score_token_stride + t2] = exp_s;
                sum_exp += exp_s;
            }

            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            for t2 in 0..kv_tokens {
                scores[score_base + t1 * score_token_stride + t2] *= inv_sum;
            }
        }
    }
}
 

/// Transforms K from [tokens, kv_heads, head_dim] to [kv_heads, head_dim, tokens]
/// the regular 2D transpose will have the same result. This is only for clarity.
/// This combines two operations:
/// 1. Swap tokens and kv_heads (standard 3D transpose)
/// 2. Swap tokens and head_dim (so tokens become the innermost dimension)
///
/// Resulting layout is optimal for batched matmul with Q: [tokens, q_heads, head_dim]
///
/// # Arguments
/// - `output`: Pre-allocated buffer of size tokens * kv_heads * head_dim
/// - `input`: Source buffer in row-major order [tokens][kv_heads][head_dim]
/// - `tokens`: Number of tokens (15 in your case)
/// - `kv_heads`: Number of KV heads (8 in your case)  
/// - `head_dim`: Dimension per head (64 in your case)
///
/// # Memory layout transformation
///
/// Input layout (row-major):
/// ┌─────────────────────────────────────┐
/// │ token0: head0[0..63], head1[0..63], ... │
/// │ token1: head0[0..63], head1[0..63], ... │
/// │ ...                                      │
/// └─────────────────────────────────────┘
///
/// Output layout (row-major):
/// ┌─────────────────────────────────────┐
/// │ head0: dim0[t0,t1..t14], dim1[t0..t14], ... │
/// │ head1: dim0[t0,t1..t14], dim1[t0..t14], ... │
/// │ ...                                          │
/// └─────────────────────────────────────┘
///
/// This is equivalent to: reshape → transpose(0,1) → transpose(1,2)
pub fn mat_transpose_3d_row_major(
    output: &mut [f32],
    input: &[f32],
    tokens: usize,
    kv_heads: usize,
    head_size: usize,
) {
    let total = tokens * kv_heads * head_size;
    assert_eq!(input.len(), total);
    assert_eq!(output.len(), total);

    // Input index: t * (kv_heads * head_size) + h * head_size + d
    // Output index: h * (head_size * tokens) + d * tokens + t
    //
    // Why this mapping?
    // - Outer loop: h (kv_heads) becomes fastest-changing in output's outer dimension
    // - Middle: d (head_size) becomes the next dimension
    // - Inner: t (tokens) becomes the innermost (contiguous in memory)
    for t in 0..tokens {
        for h in 0..kv_heads {
            for d in 0..head_size {
                let src_idx = t * (kv_heads * head_size) + h * head_size + d;
                let dst_idx = h * (head_size * tokens) + d * tokens + t;
                output[dst_idx] = input[src_idx];
            }
        }
    }
}

pub fn dequantize_mat_mul(
    output: &mut [f32],
    input: &MutableQuantizedTensorQ8,
    weights: &QuantizedTensor,
    group_size: usize,
    batch: usize,
) {
    let w = match weights {
        QuantizedTensor::Q8(ref q) => q,
        _ => panic!("LMRS sequential: only Q8_0 format is currently supported"),
    };
    let in_total = input.quant_vals.len();
    let out_total = output.len();
    let weights_total = w.quant_vals.len();

    assert!(weights_total > 0 && in_total > 0 && out_total > 0);
    assert!(batch > 0 && batch <= in_total && batch <= out_total);

    let attention_dim = in_total / batch;
    let dim = out_total / batch;
    assert_eq!(attention_dim * dim, weights_total, "weights shape mismatch");
    assert_eq!(in_total, batch * attention_dim, "input shape mismatch");
    assert_eq!(out_total, batch * dim, "output shape mismatch");
    assert!(
        in_total % group_size == 0,
        "input dim must be divisible by group_size"
    );
    assert_eq!(input.scale_factor.len(), in_total / group_size);

    for m in 0..batch {
        let in_offset = m * attention_dim;
        for j in 0..dim {
            let out_idx = m * dim + j;
            let mut sum = 0.0f32;
            let weight_col_offset = j * attention_dim;

            for group_block_start in (0..attention_dim).step_by(group_size) {
                let mut block_sum = 0i32;
                let group_end = (group_block_start + group_size).min(attention_dim);

                for n in group_block_start..group_end {
                    let w_val = w.quant_vals[weight_col_offset + n] as i32;
                    let x_val = input.quant_vals[in_offset + n] as i32;
                    block_sum += w_val * x_val;
                }

                let block_sum_weighted = (block_sum as f32)
                    * w.scale_factor[(weight_col_offset + group_block_start) / group_size]
                    * input.scale_factor[(in_offset + group_block_start) / group_size];

                sum += block_sum_weighted;
            }
            output[out_idx] = sum;
        }
    }
}

pub fn rope_scalar(shape: &TransformerShape, sq: &mut [f32], k: &mut [f32], pos: u32) {
    // RoPE (Rotary Position Embedding) - Llama-specific YARN-style frequency scaling
    //
    // WHAT IS RoPE?
    // RoPE encodes position information by rotating query and key vectors in 2D subspaces.
    // Each dimension pair (i, i+head_size/2) is treated as a complex number and rotated
    // by an angle proportional to the token position.
    //
    // WHY Llama IS SPECIAL:
    // Standard RoPE uses a fixed base frequency (rope_theta) that scales linearly with position.
    // Llama introduces "frequency scaling" to handle long contexts better (>32K tokens).
    //
    // THE Llama 3 FORMULA:
    // 1. Base frequency: freq = 1 / theta^(2i/head_size) where i is dimension index
    // 2. YARN-style adjustment:
    //    - factor = 32.0 (Llama's attention scaling factor)
    //    - low_freq_factor = 1.0, high_freq_factor = 4.0
    //    - If wavelen > old_context_len: freq /= 32 (boost long-range)
    //    - If between boundaries: smooth interpolation (blend)
    //    - Otherwise: keep original frequency
    //
    // SIMPLIFIED (no Rayon for WebAssembly compatibility):
    // - Iterate over each KV head and Q head
    // - For each 2D rotation pair (j, j+half_head_size), apply complex rotation
    // - Only apply rotation to K heads that exist (GQA: some Q heads share same K head)

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

    // Iterate over each dimension pair (j, j+half_head_size) in each head
    for j in 0..half_head_size {
        let head_dim = j * 2;
        let mut freq = rope_theta.powf(-(head_dim as f32) / head_size as f32);

        // YARN frequency adjustment
        let wavelen = (2.0 * std::f32::consts::PI) / freq;
        if wavelen > low_freq_wavelen {
            freq /= rope_factor;
        } else if wavelen <= low_freq_wavelen && wavelen >= high_freq_wavelen {
            let smooth_factor = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            freq = (1.0 - smooth_factor) * freq / rope_factor + smooth_factor * freq;
        }

        // Rotation angle for this dimension pair
        let val = (pos as f32) * freq;
        let fcr = val.cos();
        let fci = val.sin();

        // Apply rotation to Q heads
        for attention_head in 0..n_heads {
            let offset = attention_head * head_size + j;

            // Rotate dimension pair (offset, offset+half_head_size) as complex number
            let v0q = sq[offset];
            let v1q = sq[offset + half_head_size];
            sq[offset] = v0q * fcr - v1q * fci;
            sq[offset + half_head_size] = v0q * fci + v1q * fcr;
        }

        // Apply rotation to K heads (only if this head index exists in KV heads)
        for kv_idx in 0..n_kv_heads {
            let offset = kv_idx * head_size + j;

            let v0k = k[offset];
            let v1k = k[offset + half_head_size];
            k[offset] = v0k * fcr - v1k * fci;
            k[offset + half_head_size] = v0k * fci + v1k * fcr;
        }
    }
}
