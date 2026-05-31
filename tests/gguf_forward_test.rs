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


use yallma3_llm::TransformerLlamaGguf;
use yallma3_llm::Transformer;

#[test]
fn test_gguf_forward_pass() {
    let model = TransformerLlamaGguf::load_model(
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        u32::MAX,
    ).unwrap();
    let mut session = model.create_session();
    let logits = model.forward_x(&[model.bos_token()], 0, &mut session);

    assert_eq!(logits.len(), model.vocab_size(), "logits length mismatch");

    for &v in logits.iter().take(10) {
        assert!(!v.is_nan(), "logit is NaN");
    }

    let sum: f32 = logits.iter().take(100).sum();
    assert!(sum != 0.0, "logits are all zero");

    println!("Forward pass OK, first 10 logits: {:?}", &logits[..10]);
}

#[test]
fn test_gguf_prefill_and_generate() {
    let model = TransformerLlamaGguf::load_model(
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        u32::MAX,
    ).unwrap();
    let mut session = model.create_session();

    // Simulate a 3-token prompt
    let prompt = &[model.bos_token(), 9125, model.bos_token()];
    let logits = model.forward_x(prompt, 0, &mut session);
    session.pos += prompt.len() as u32;

    // Find argmax
    let first = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    println!("After prefill (3 tokens), argmax token: {} (logit: {:?})", first, logits[first]);
    println!("  token 128006 logit: {:?}", logits[128006]);

    // Generation step 1
    let gen_pos = session.pos;
    let logits2 = model.forward_x(&[first as u32], gen_pos, &mut session);
    session.pos += 1;

    let second = logits2.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    println!("After gen step 1, argmax token: {} (logit: {:?})", second, logits2[second]);
    println!("  token 128006 logit: {:?}", logits2[128006]);

    assert!(logits.iter().all(|v| v.is_finite()), "Prefill logits should be finite");
    assert!(logits2.iter().all(|v| v.is_finite()), "Gen step 1 logits should be finite");
    println!("Short prompt test OK");
}

#[test]
fn test_gguf_long_prompt_loop() {
    let model = TransformerLlamaGguf::load_model(
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        u32::MAX,
    ).unwrap();
    let mut session = model.create_session();
    let tokenizer = model.tokenizer().unwrap();

    // Build prompt using same flow as main.rs (with generation prompt prefix)
    let prompt = tokenizer.apply_turn_template("hi", true);

    let logits = model.forward_x(&prompt, 0, &mut session);
    session.pos += prompt.len() as u32;

    // Top 5 tokens after prefill
    let mut sorted: Vec<(usize, &f32)> = logits.iter().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("After prefill top 5:");
    for (tok, logit) in sorted.iter().take(5) {
        println!("  token {}: logit {}", tok, logit);
    }

    let first = sorted[0].0 as u32;

    // Generation step 0
    let logits2 = model.forward_x(&[first], prompt.len() as u32, &mut session);
    session.pos += 1;

    let mut sorted2: Vec<(usize, &f32)> = logits2.iter().enumerate().collect();
    sorted2.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("After gen step 0 top 5:");
    for (tok, logit) in sorted2.iter().take(5) {
        println!("  token {}: logit {}", tok, logit);
    }

    // Verify neither step predicts 128006 (the prompt already has the assistant header)
    assert!(sorted[0].0 != 128006,
            "Prefill predicts 128006 but assistant header is already in prompt");
    assert!(sorted2[0].0 != 128006,
            "Gen step 0 predicts 128006 (looping)");
    println!("Long prompt test OK (with generation prompt prefix)");
}
