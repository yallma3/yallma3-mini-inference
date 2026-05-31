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


use crate::quantization::{MutableQuantizedTensorQ8, QuantType};
use crate::tokenizer::Tokenizer;
use crate::weight_type::QuantizedTensor;

#[derive(Debug, Clone, Copy)]
pub enum ModelFamily {
    Qwen2 = 0,
    Llama = 1,
}

#[derive(Debug)]
pub struct TransformerShape {
    pub dimension: u32,
    pub hidden_dimension: u32,
    pub n_heads: u32,
    pub head_size: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub n_layers: u32,
    pub ctx_len: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub q_type: QuantType,
    pub group_size: u32,
    pub rope_freq_scale: f32,
    pub rope_ext_factor: f32,
    pub rope_attn_factor: f32,
    pub rope_original_ctx_len: u32,
    pub attention_scale: f32,
}

#[derive(Debug)]
#[deprecated(note = "Use InferenceSession instead. TransformerWorkspace will be removed in a future version.")]
pub struct TransformerWorkspace {
    pub xb: Vec<f32>,
    pub xq: Option<MutableQuantizedTensorQ8>,
    pub logits: Vec<f32>,

    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

#[derive(Debug)]
pub struct ModelWeights {
    pub token_embedding: Vec<f32>,
    pub token_embedding_quant: Option<QuantizedTensor>,

    pub wq: Vec<QuantizedTensor>,
    pub wk: Vec<QuantizedTensor>,
    pub wv: Vec<QuantizedTensor>,
    pub wo: Vec<QuantizedTensor>,

    pub w_rms_att: Vec<Vec<f32>>,

    pub w1: Vec<QuantizedTensor>,
    pub w2: Vec<QuantizedTensor>,
    pub w3: Vec<QuantizedTensor>,

    pub w_rms_post_att: Vec<Vec<f32>>,

    pub w_rms_final: Vec<f32>,
}

/// Per-session state for multi-turn conversation inference.
/// Contains the KV cache and current position — the persistent state carried across tokens.
/// Temp buffers (logits, xb, xq) are allocated locally in forward_x.
#[derive(Debug)]
pub struct InferenceSession {
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub pos: u32,
}

impl InferenceSession {
    pub fn new(shape: &TransformerShape) -> Self {
        let kv_dim = (shape.n_kv_heads * shape.head_size) as usize;
        let cache_size = shape.n_layers as usize * shape.ctx_len as usize * kv_dim;

        Self {
            key_cache: vec![0.0; cache_size],
            value_cache: vec![0.0; cache_size],
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);
        self.pos = 0;
    }
}

#[derive(Debug)]
pub struct TransformerBase {
    pub model_info: String,
    pub model_family: Option<ModelFamily>,
    pub shape: Option<TransformerShape>,
    pub weights: Option<ModelWeights>,
    pub tokenizer: Option<Tokenizer>,
}

impl TransformerBase {
    pub fn to_string(&self) -> String {
        format!(
            "TransformerBase {{\n\
             model_info: {},\n\
             model_family: {:?},\n\
             shape: {:?},\n\
             weights: {:?}\n\
             }}",
            self.model_info, self.model_family, self.shape, self.weights
        )
    }

    pub fn get_embeddings(&self, token: u32) -> Vec<f32> {
        let shape = self.shape.as_ref().expect("Shape not initialized");
        let weights = self.weights.as_ref().expect("Weights not initialized");

        let dim = shape.dimension as usize;
        let mut embedding = vec![0.0; dim];
        let token_embed_idx = (token as usize) * dim;
        if token_embed_idx + dim <= weights.token_embedding.len() {
            embedding
                .copy_from_slice(&weights.token_embedding[token_embed_idx..token_embed_idx + dim]);
        }
        embedding
    }

    pub fn get_embeddings_batch(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let shape = self.shape.as_ref().expect("Shape not initialized");
        let weights = self.weights.as_ref().expect("Weights not initialized");
        let dim = shape.dimension as usize;
        let mut embeddings = Vec::with_capacity(tokens.len());
        for token in tokens {
            let token_embed_idx = (*token as usize) * dim;
            if token_embed_idx + dim <= weights.token_embedding.len() {
                let mut embedding = vec![0.0; dim];
                embedding.copy_from_slice(
                    &weights.token_embedding[token_embed_idx..token_embed_idx + dim],
                );
                embeddings.push(embedding);
            } else {
                embeddings.push(vec![0.0; dim]);
            }
        }
        embeddings
    }
}
