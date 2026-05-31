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


use yallma3_llm::TransformerLlamaSt;
use yallma3_llm::Transformer;

#[test]
fn test_st_forward_pass() {
    let model = TransformerLlamaSt::load_model("models/llama_q8_st/model.safetensors", u32::MAX).unwrap();
    let mut session = model.create_session();

    let logits = model.forward_x(&[model.bos_token()], 0, &mut session);
    assert_eq!(logits.len(), model.vocab_size());
    for &v in logits.iter().take(10) {
        assert!(!v.is_nan(), "logit is NaN");
    }
    let sum: f32 = logits.iter().take(100).sum();
    assert!(sum != 0.0, "logits are all zero");
    println!("Forward pass OK, first 10 logits: {:?}", &logits[..10]);
}
