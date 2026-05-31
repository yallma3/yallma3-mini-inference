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
use crate::quantization::{MutableQuantizedTensorQ8, QuantType};
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use crate::weight_type::{QuantizedTensor, WeightFormat};
use memmap2::Mmap;
use std::convert::TryInto;
use std::fs::File;
use std::mem::size_of;
use std::path::Path;

const LLAMA_BOS: u32 = 1;
const LLAMA_EOS: u32 = 2;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
struct TransformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    n_kv_heads: u32,
    pub vocab_size: u32,
    seq_len: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    q_type: QuantType,
    pub model_type: ModelFamily,
    group_size: u32,
    pub multimodal: bool,
}

pub struct TransformerLlamaLmrs {
    base: TransformerBase,
    weight_format: WeightFormat,
}

impl TransformerLlamaLmrs {
    pub fn load_model_with_override(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, fmt_override)
    }

    fn load_model_inner(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = Path::new(model_path);
        if !model_path.exists() {
            return Err(format!("Model file does not exist: {}", model_path.display()).into());
        }

        let tokenizer = Tokenizer::new(&model_path.with_file_name("tokenizer.bin"))?;

        println!("Loading Llama LMRS model from: {}", model_path.display());
        let file = File::open(model_path).expect("Error opening model file");
        let raw_model = unsafe { Mmap::map(&file).expect("MMap failed") };

        assert_eq!(
            raw_model[0..4],
            [0x6c, 0x6d, 0x72, 0x73],
            "Model not in lm.rs format."
        );
        let version = u32::from_le_bytes(raw_model[4..8].try_into().unwrap());
        print!("LMRS model version: {}\n", version);

        let config_slice = &raw_model[8..55];
        if config_slice.len() < std::mem::size_of::<TransformerArgs>() {
            return Err(
                "Invalid model configuration: insufficient data for TransformerArgs".into(),
            );
        }
        let cfg_ptr = config_slice.as_ptr() as *const TransformerArgs;
        let cfg = unsafe { *cfg_ptr };

        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let n_heads = cfg.n_heads;
        let n_layers = cfg.n_layers;
        let seq_len = cfg.seq_len;
        let vocab_size = cfg.vocab_size;
        let rms_norm_eps = cfg.rms_norm_eps;
        let rope_theta = cfg.rope_theta;
        let q_type = cfg.q_type;
        let group_size = cfg.group_size;

        let weight_format = match fmt_override {
            Some(f) => {
                if f != WeightFormat::Q8_0 {
                    eprintln!("Warning: LMRS format only supports Q8_0, ignoring override to {:?}", f);
                }
                WeightFormat::Q8_0
            }
            None => WeightFormat::Q8_0,
        };

        if dim == 0 || hidden_dim == 0 || n_heads == 0 || n_layers == 0 {
            return Err("Invalid model configuration: critical dimensions cannot be zero".into());
        }
        if seq_len == 0 || vocab_size == 0 {
            return Err(
                "Invalid model configuration: sequence length and vocab size must be positive"
                    .into(),
            );
        }

        let ctx_len = seq_len.min(max_ctx_len);
        if max_ctx_len < u32::MAX && seq_len > max_ctx_len {
            println!(
                "Warning: Model context length {} exceeds maximum ({}), capping.",
                seq_len, max_ctx_len
            );
        }

        let shape = TransformerShape {
            dimension: dim,
            hidden_dimension: hidden_dim,
            n_heads,
            head_size: cfg.head_size,
            n_kv_heads: cfg.n_kv_heads,
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

        if shape.q_type != QuantType::None {
            println!(
                "Using {:?} quantization with group size {}",
                shape.q_type, shape.group_size
            );
        }

        let mut offset = 256;
        let token_embedding_q8_size = shape.dimension as usize * shape.vocab_size as usize;

        let token_embedding_raw = {
            let qt = crate::util::load_q8_tenstor_lmrs(
                &raw_model, offset,
                shape.vocab_size as usize, shape.dimension as usize, shape.group_size as usize,
            );
            offset += qt.1;
            qt.0
        };

        let mut emb_tab: Vec<f32> = vec![0.0; token_embedding_q8_size];
        for (i, value) in emb_tab.iter_mut().enumerate().take(token_embedding_q8_size) {
            *value = (token_embedding_raw.quant_vals[i] as f32)
                * token_embedding_raw.scale_factor[i / (shape.group_size as usize)];
        }

        let mut model_weights = ModelWeights {
            token_embedding: emb_tab,
            token_embedding_quant: Some(QuantizedTensor::Q8(token_embedding_raw)),
            w_rms_att: vec![vec![0.0; shape.dimension as usize]; shape.n_layers as usize],
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            w_rms_post_att: vec![vec![0.0; shape.dimension as usize]; shape.n_layers as usize],
            w1: Vec::new(),
            w2: Vec::new(),
            w3: Vec::new(),
            w_rms_final: vec![0.0; shape.dimension as usize],
        };

        for layer in 0..shape.n_layers {
            model_weights.w_rms_att[layer as usize].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + (shape.dimension as usize * size_of::<f32>())]
                        .as_ptr() as *const f32,
                    shape.dimension as usize,
                )
            });
            offset += shape.dimension as usize * size_of::<f32>();
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (shape.n_heads * shape.head_size * shape.dimension) as usize],
                scale_factor: vec![
                    0.0;
                    (shape.n_heads * shape.head_size * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let wq_size = (shape.n_heads * shape.head_size * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wq_size].as_ptr() as *const i8,
                    wq_size,
                )
            });
            offset += wq_size * size_of::<i8>();
            let wq_scale_size =
                (shape.n_heads * shape.head_size * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wq_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    wq_scale_size,
                )
            });
            offset += wq_scale_size * size_of::<f32>();
            model_weights.wq.push(QuantizedTensor::Q8(qt));
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![
                    0;
                    (shape.n_kv_heads * shape.head_size * shape.dimension) as usize
                ],
                scale_factor: vec![
                    0.0;
                    (shape.n_kv_heads * shape.head_size * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let wk_size = (shape.n_kv_heads * shape.head_size * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wk_size].as_ptr() as *const i8,
                    wk_size,
                )
            });
            offset += wk_size * size_of::<i8>();
            let wk_scale_size =
                (shape.n_kv_heads * shape.head_size * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wk_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    wk_scale_size,
                )
            });
            offset += wk_scale_size * size_of::<f32>();
            model_weights.wk.push(QuantizedTensor::Q8(qt));
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![
                    0;
                    (shape.n_kv_heads * shape.head_size * shape.dimension) as usize
                ],
                scale_factor: vec![
                    0.0;
                    (shape.n_kv_heads * shape.head_size * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let wv_size = (shape.n_kv_heads * shape.head_size * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wv_size].as_ptr() as *const i8,
                    wv_size,
                )
            });
            offset += wv_size * size_of::<i8>();
            let wv_scale_size =
                (shape.n_kv_heads * shape.head_size * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wv_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    wv_scale_size,
                )
            });
            offset += wv_scale_size * size_of::<f32>();
            model_weights.wv.push(QuantizedTensor::Q8(qt));
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (shape.n_heads * shape.head_size * shape.dimension) as usize],
                scale_factor: vec![
                    0.0;
                    (shape.n_heads * shape.head_size * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let wo_size = (shape.n_heads * shape.head_size * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wo_size].as_ptr() as *const i8,
                    wo_size,
                )
            });
            offset += wo_size * size_of::<i8>();
            let wo_scale_size =
                (shape.n_heads * shape.head_size * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + wo_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    wo_scale_size,
                )
            });
            offset += wo_scale_size * size_of::<f32>();
            model_weights.wo.push(QuantizedTensor::Q8(qt));
        }

        for layer in 0..shape.n_layers {
            model_weights.w_rms_post_att[layer as usize].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + (shape.dimension as usize * size_of::<f32>())]
                        .as_ptr() as *const f32,
                    shape.dimension as usize,
                )
            });
            offset += shape.dimension as usize * size_of::<f32>();
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (shape.hidden_dimension * shape.dimension) as usize],
                scale_factor: vec![
                    0.0;
                    (shape.hidden_dimension * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let w1_size = (shape.hidden_dimension * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w1_size].as_ptr() as *const i8,
                    w1_size,
                )
            });
            offset += w1_size * size_of::<i8>();
            let w1_scale_size =
                (shape.hidden_dimension * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w1_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    w1_scale_size,
                )
            });
            offset += w1_scale_size * size_of::<f32>();
            model_weights.w1.push(QuantizedTensor::Q8(qt));
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (shape.hidden_dimension * shape.dimension) as usize],
                scale_factor: vec![
                    0.0;
                    (shape.hidden_dimension * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let w2_size = (shape.hidden_dimension * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w2_size].as_ptr() as *const i8,
                    w2_size,
                )
            });
            offset += w2_size * size_of::<i8>();
            let w2_scale_size =
                (shape.hidden_dimension * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w2_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    w2_scale_size,
                )
            });
            offset += w2_scale_size * size_of::<f32>();
            model_weights.w2.push(QuantizedTensor::Q8(qt));
        }

        for _layer in 0..shape.n_layers {
            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0; (shape.hidden_dimension * shape.dimension) as usize],
                scale_factor: vec![
                    0.0;
                    (shape.hidden_dimension * shape.dimension / shape.group_size)
                        as usize
                ],
            };
            let w3_size = (shape.hidden_dimension * shape.dimension) as usize;
            qt.quant_vals.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w3_size].as_ptr() as *const i8,
                    w3_size,
                )
            });
            offset += w3_size * size_of::<i8>();
            let w3_scale_size =
                (shape.hidden_dimension * shape.dimension / shape.group_size) as usize;
            qt.scale_factor.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    raw_model[offset..offset + w3_scale_size * size_of::<f32>()].as_ptr()
                        as *const f32,
                    w3_scale_size,
                )
            });
            offset += w3_scale_size * size_of::<f32>();
            model_weights.w3.push(QuantizedTensor::Q8(qt));
        }

        model_weights.w_rms_final.copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                raw_model[offset..offset + (shape.dimension as usize * size_of::<f32>())].as_ptr()
                    as *const f32,
                shape.dimension as usize,
            )
        });
        offset += shape.dimension as usize * size_of::<f32>();
        println!("Model loaded successfully. Total size: {} bytes", offset);

        Ok(Self {
            weight_format,
            base: TransformerBase {
                model_info: format!("Llama LMRS model loaded from {}", model_path.display()),
                model_family: ModelFamily::Llama.into(),
                shape: Some(shape),
                weights: Some(model_weights),
                tokenizer: Some(tokenizer),
            },
        })
    }
}

impl Transformer for TransformerLlamaLmrs {
    fn load_model(model_path: &str, max_ctx_len: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, None)
    }

    fn name(&self) -> &str {
        "TransformerLlamaLmrs"
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

impl TransformerLlamaLmrs {
    pub fn to_string(&self) -> String {
        self.base.to_string()
    }
}



