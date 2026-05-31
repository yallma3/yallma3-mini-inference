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


use crate::base_transformer::InferenceSession;
use crate::tokenizer::Tokenizer;

/// Trait for transformer-based models
pub trait Transformer: Send + Sync {
    fn load_model(model_path: &str, max_ctx_len: u32) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn name(&self) -> &str;
    fn vocab_size(&self) -> usize;

    /// Get embedding for a token
    ///
    /// # Arguments
    /// * `token` - The input token ID
    ///
    /// # Returns
    /// Embedding vector for the token
    fn get_embeddings(&self, token: u32) -> Vec<f32>;

    /// Get embeddings for a batch of tokens
    ///
    /// # Arguments
    /// * `tokens` - A slice of token IDs
    ///
    /// # Returns
    /// A vector of embedding vectors, one for each token
    fn get_embeddings_batch(&self, tokens: &[u32]) -> Vec<Vec<f32>>;

    /// Forward pass for a batch of tokens at a given start position
    ///
    /// # Arguments
    /// * `tokens` - The input token IDs
    /// * `start_pos` - The position of the first token in the sequence
    /// * `session` - Per-session state containing the KV cache
    ///
    /// # Returns
    /// Logits for the next token prediction
    fn forward_x(&self, tokens: &[u32], start_pos: u32, session: &mut InferenceSession) -> Vec<f32>;

    /// Create a new inference session with KV cache sized for this model
    fn create_session(&self) -> InferenceSession;

    /// Get maximum context length
    fn max_ctx_len(&self) -> u32;

    /// Get EOS token id
    fn eos_token(&self) -> u32;

    /// Get BOS token id
    fn bos_token(&self) -> u32;

    /// Get a reference to the tokenizer, if loaded
    fn tokenizer(&self) -> Option<&Tokenizer>;
}
