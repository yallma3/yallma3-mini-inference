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


use crate::base_transformer::{
    InferenceSession, ModelFamily, ModelWeights, TransformerBase, TransformerShape,
};
use crate::quantization::QuantType;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use crate::weight_type::{QuantizedTensor, WeightFormat};
use memmap2::Mmap;
use rayon::prelude::*;
use std::convert::TryInto;
use std::fs::File;
use std::path::Path;

const LLAMA_BOS: u32 = 1;
const LLAMA_EOS: u32 = 2;

pub struct TransformerLlamaSt {
    base: TransformerBase,
    weight_format: WeightFormat,
}

impl TransformerLlamaSt {
    pub fn load_model_with_override(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, fmt_override)
    }

    fn load_model_inner(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = Path::new(model_path);
        let model_dir = model_path.parent().unwrap();

        // --- Phase 2 start: read config.json ---
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
        #[cfg(feature = "debug_prints")]
        println!("rope_theta={}, max_seq_len={}", rope_theta, max_seq_len);
        let (q_type, weight_format, group_size) = match fmt_override {
            Some(f) => match f {
                WeightFormat::Q8_0 => (QuantType::Q8_0, WeightFormat::Q8_0,
                    config.get("quantization").and_then(|q| q.get("group_size")).and_then(|v| v.as_u64()).unwrap_or(32) as u32),
                WeightFormat::Q4_0 => (QuantType::Q4_0, WeightFormat::Q4_0,
                    config.get("quantization").and_then(|q| q.get("group_size")).and_then(|v| v.as_u64()).unwrap_or(32) as u32),
                WeightFormat::F16 => (QuantType::F16, WeightFormat::F16, 32),
            },
            None => {
                let q = config.get("quantization");
                match q {
                    // No quantization section at all → raw F16 weights (HuggingFace format)
                    None => (QuantType::F16, WeightFormat::F16, 32),
                    Some(q_obj) => {
                        // Has a quantization section – check quant_type
                        match q_obj.get("quant_type").and_then(|v| v.as_str()) {
                            Some("Q4_0") => {
                                let gs = q_obj.get("group_size").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
                                (QuantType::Q4_0, WeightFormat::Q4_0, gs)
                            }
                            // Default to Q8_0 (compatible with original ST format that
                            // had "quantization": {"group_size": N} without quant_type)
                            _ => {
                                let gs = q_obj.get("group_size").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
                                (QuantType::Q8_0, WeightFormat::Q8_0, gs)
                            }
                        }
                    }
                }
            }
        };

        #[cfg(feature = "debug_prints")]
        println!("Config loaded: dim={}, layers={}, heads={}, kv_heads={}, fmt={:?}",
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

        // --- Step 3: Safetensors header parser ---
        // Phase 1: Read the binary safetensors header to get byte offsets for every tensor.
        // The mmap approach allows zero-copy access to the weight data.
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Safetensors format: 8 bytes header_len (u64 LE) | JSON header | raw data
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&mmap[8..8 + header_len])?;
        let header: serde_json::Value = serde_json::from_str(header_str)?;

        // Build a lookup map from tensor name to its metadata (dtype, shape, byte offset, size)
        // This is Phase 1's output — the raw tensor info that Phase 2 uses to load weights.
        struct TensorInfo {
            dtype: String,
            shape: Vec<usize>,
            offset: usize,
        }

        let data_start = 8 + header_len;
        let mut tensors: std::collections::HashMap<String, TensorInfo> = std::collections::HashMap::new();

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

        #[cfg(feature = "debug_prints")]
        println!("Found {} tensors in safetensors file", tensors.len());

        // --- Step 4: F16 → F32 loader helper ---
        // Used for loading norm weights stored as float16 in the safetensors file.
        fn load_f16_vec(data: &[u8], start: usize, len: usize) -> Vec<f32> {
            let mut result = vec![0.0f32; len];
            for i in 0..len {
                let bits = u16::from_le_bytes(data[start + i * 2..][..2].try_into().unwrap());
                result[i] = half::f16::from_bits(bits).to_f32();
            }
            result
        }

        // --- Step 4: Load model.norm.weight (final RMS norm) ---
        let dim_usize = dim as usize;
        let norm_weight_info = &tensors["model.norm.weight"];
        assert_eq!(norm_weight_info.dtype, "F16");
        assert_eq!(norm_weight_info.shape[0], dim_usize);
        let w_rms_final = load_f16_vec(&mmap, norm_weight_info.offset, dim_usize);
        #[cfg(feature = "debug_prints")]
        println!("model.norm.weight first 4 values: {:?}", &w_rms_final[..4]);

        // --- Step 5: Load all layer norm weights in parallel ---
        let n_layers_usize = n_layers as usize;
        let norm_names: Vec<(String, String)> = (0..n_layers_usize)
            .map(|i| {
                (format!("model.layers.{}.input_layernorm.weight", i),
                 format!("model.layers.{}.post_attention_layernorm.weight", i))
            })
            .collect();
        let norm_results: Vec<(Vec<f32>, Vec<f32>)> = norm_names.par_iter()
            .map(|(att_name, post_name)| {
                let att = load_f16_vec(&mmap, tensors[att_name.as_str()].offset, dim_usize);
                let post = load_f16_vec(&mmap, tensors[post_name.as_str()].offset, dim_usize);
                (att, post)
            })
            .collect();
        let mut w_rms_att = Vec::with_capacity(n_layers_usize);
        let mut w_rms_post_att = Vec::with_capacity(n_layers_usize);
        for (att, post) in norm_results {
            w_rms_att.push(att);
            w_rms_post_att.push(post);
        }
        #[cfg(feature = "debug_prints")]
        println!("Layer 0 input_norm first 4: {:?}", &w_rms_att[0][..4]);

        // --- Step 6 & 7: Load all tensors ---
        let vocab_size_usize = vocab_size as usize;
        let hidden_dim_usize = hidden_dim as usize;
        let n_heads_usize = n_heads as usize;
        let head_size_usize = head_size as usize;
        let n_kv_heads_usize = n_kv_heads as usize;
        let q_rows = n_heads_usize * head_size_usize;
        let kv_rows = n_kv_heads_usize * head_size_usize;
        let gs = group_size as usize;

        // --- Token embedding ---
        let token_embedding_quant: Option<QuantizedTensor>;
        let token_embedding: Vec<f32>;
        if weight_format == WeightFormat::F16 {
            let ti = &tensors["model.embed_tokens.weight"];
            assert_eq!(ti.dtype, "F16", "Expected F16 token embedding");
            let n = ti.shape[0] * ti.shape[1];
            let mut emb = vec![0.0f32; n];
            for i in 0..n {
                let bits = u16::from_le_bytes(mmap[ti.offset + i * 2..][..2].try_into().unwrap());
                emb[i] = half::f16::from_bits(bits).to_f32();
            }
            token_embedding_quant = Some(QuantizedTensor::F16(emb.clone()));
            token_embedding = emb;
        } else {
            let wt = &tensors["model.embed_tokens.weight"];
            let st = &tensors["model.embed_tokens.scales"];
            let bt = &tensors["model.embed_tokens.biases"];
            let embed_qt = crate::util::load_q8_tensor_st(
                &mmap, vocab_size_usize, dim_usize, gs,
                wt.offset, st.offset, bt.offset,
            );
            token_embedding_quant = Some(QuantizedTensor::Q8(embed_qt));
            let embed_ref = match token_embedding_quant.as_ref().unwrap() {
                QuantizedTensor::Q8(ref q) => q,
                _ => unreachable!(),
            };
            let mut emb = vec![0.0; vocab_size_usize * dim_usize];
            let gs_embed = gs;
            let qv = &embed_ref.quant_vals;
            let sf = &embed_ref.scale_factor;
            emb.par_chunks_mut(gs_embed).enumerate().for_each(|(g, chunk)| {
                let scale = sf[g];
                let base = g * gs_embed;
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = qv[base + j] as f32 * scale;
                }
            });
            token_embedding = emb;
        }

        #[cfg(feature = "debug_prints")]
        println!("token_embd[0] first 8: {:?}", &token_embedding[..8]);
        #[cfg(feature = "debug_prints")]
        println!("token_embd[1000] first 8: {:?}", &token_embedding[1000*dim_usize..1000*dim_usize+8]);
        #[cfg(feature = "debug_prints")]
        println!("ST vocab_size={}, token_embedding len={}", vocab_size, token_embedding.len());

        // --- Helper: load F16 tensor from safetensors and return as QuantizedTensor::F16 ---
        fn load_f16_tensor_st(mmap: &[u8], ti: &TensorInfo) -> QuantizedTensor {
            assert_eq!(ti.dtype, "F16", "Expected F16 tensor, got {}", ti.dtype);
            let n: usize = ti.shape.iter().product();
            let mut result = vec![0.0f32; n];
            for i in 0..n {
                let bits = u16::from_le_bytes(mmap[ti.offset + i * 2..][..2].try_into().unwrap());
                result[i] = half::f16::from_bits(bits).to_f32();
            }
            QuantizedTensor::F16(result)
        }

        // --- Load all per-layer tensors in parallel ---
        struct LayerWeights {
            wq: QuantizedTensor,
            wk: QuantizedTensor,
            wv: QuantizedTensor,
            wo: QuantizedTensor,
            w1: QuantizedTensor,
            w2: QuantizedTensor,
            w3: QuantizedTensor,
        }

        let is_quantized = weight_format != WeightFormat::F16;
        let layer_tensors: Vec<LayerWeights> = (0..n_layers_usize).into_par_iter().map(|i| {
            let prefix = format!("model.layers.{}", i);

            let load_q = |proj: &str, rows: usize, cols: usize| {
                if is_quantized {
                    let base = format!("{}.{}", prefix, proj);
                    let w = &tensors[&format!("{}.weight", base)];
                    let s = &tensors[&format!("{}.scales", base)];
                    let b = &tensors[&format!("{}.biases", base)];
                    QuantizedTensor::Q8(crate::util::load_q8_tensor_st(&mmap, rows, cols, gs, w.offset, s.offset, b.offset))
                } else {
                    let tensor_name = format!("{}.{}.weight", prefix, proj);
                    let ti = &tensors[&tensor_name];
                    load_f16_tensor_st(&mmap, ti)
                }
            };

            LayerWeights {
                wq: load_q("self_attn.q_proj", q_rows, dim_usize),
                wk: load_q("self_attn.k_proj", kv_rows, dim_usize),
                wv: load_q("self_attn.v_proj", kv_rows, dim_usize),
                wo: load_q("self_attn.o_proj", dim_usize, q_rows),
                w1: load_q("mlp.gate_proj", hidden_dim_usize, dim_usize),
                w2: load_q("mlp.down_proj", dim_usize, hidden_dim_usize),
                w3: load_q("mlp.up_proj", hidden_dim_usize, dim_usize),
            }
        }).collect();

        let mut wq = Vec::with_capacity(n_layers_usize);
        let mut wk = Vec::with_capacity(n_layers_usize);
        let mut wv = Vec::with_capacity(n_layers_usize);
        let mut wo = Vec::with_capacity(n_layers_usize);
        let mut w1 = Vec::with_capacity(n_layers_usize);
        let mut w2 = Vec::with_capacity(n_layers_usize);
        let mut w3 = Vec::with_capacity(n_layers_usize);
        for lt in layer_tensors {
            wq.push(lt.wq);
            wk.push(lt.wk);
            wv.push(lt.wv);
            wo.push(lt.wo);
            w1.push(lt.w1);
            w2.push(lt.w2);
            w3.push(lt.w3);
        }

        #[cfg(feature = "debug_prints")]
        println!("Loaded {} layers with 7 Q8 tensors each + 1 embedding ({} total Q8 tensors)",
                 n_layers_usize, n_layers_usize * 7 + 1);

        let model_weights = ModelWeights {
            token_embedding,
            token_embedding_quant,
            wq, wk, wv, wo,
            w_rms_att,
            w1, w2, w3,
            w_rms_post_att,
            w_rms_final,
        };

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_json(&tokenizer_path)?
        } else {
            Tokenizer::new(&model_dir.join("tokenizer.bin"))?
        };

        Ok(Self {
            weight_format,
            base: TransformerBase {
                model_info: format!("Llama Safetensors model loaded from {}", model_path.display()),
                model_family: Some(ModelFamily::Llama),
                shape: Some(shape),
                weights: Some(model_weights),
                tokenizer: Some(tokenizer),
            },
        })
    }
}

impl Transformer for TransformerLlamaSt {
    fn load_model(model_path: &str, max_ctx_len: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, None)
    }

    fn name(&self) -> &str {
        "TransformerLlamaSt"
    }

    fn vocab_size(&self) -> usize {
        self.base
            .shape
            .as_ref()
            .map(|s| s.vocab_size as usize)
            .unwrap_or(0)
    }

    fn forward_x(&self, tokens: &[u32], start_pos: u32, session: &mut InferenceSession) -> Vec<f32> {
        let shape = self.base.shape.as_ref().expect("Shape not initialized");
        let weights = self.base.weights.as_ref().expect("Weights not initialized");
        crate::forward::llama_forward_x(
            self.weight_format,
            crate::util::rope,
            tokens, start_pos, session,
            shape, weights,
            &|t| self.get_embeddings_batch(t),
        )
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

impl TransformerLlamaSt {
    pub fn to_string(&self) -> String {
        self.base.to_string()
    }
}
