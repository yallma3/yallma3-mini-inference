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


use std::io;
use yallma3_llm::util::get_simd_level;
use yallma3_llm::weight_type::WeightFormat;
use yallma3_llm::{
    configure_rayon, load_transformer, print_parallelism_config, InferenceSession,
    ParallelismConfig, Sampler,
};

fn get_or_create_session<'a>(
    sessions: &'a mut Vec<InferenceSession>,
    idx: usize,
    transformer: &dyn yallma3_llm::Transformer,
) -> &'a mut InferenceSession {
    while sessions.len() <= idx {
        sessions.push(transformer.create_session());
    }
    &mut sessions[idx]
}

fn validate_args(args: &[String]) -> Vec<String> {
    let     valid_args = [
        "--model_path",
        "--model-type",
        "--threads",
        "--parallelism",
        "--no-parallel",
        "--sequential",
        "--max_ctx_len",
        "--max-tokens",
        "--help",
    ];

    let unknown: Vec<String> = args
        .iter()
        .filter(|arg| {
            if arg.starts_with("--") && !arg.starts_with("---") {
                !valid_args.contains(&arg.as_str())
            } else {
                false
            }
        })
        .cloned()
        .collect();

    unknown
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help") {
        print_usage();
        std::process::exit(0);
    }

    let unknown_args = validate_args(&args);
    if !unknown_args.is_empty() {
        eprintln!("Error: Unknown argument(s): {}", unknown_args.join(", "));
        eprintln!();
        print_usage();
        std::process::exit(1);
    }

    println!("=== Hardware Detection ===");
    let simd_level = get_simd_level();
    match simd_level {
        yallma3_llm::util::SimdLevel::Avx512 => {
            println!("SIMD: AVX-512 (512-bit registers, fastest)")
        }
        yallma3_llm::util::SimdLevel::Avx2 => println!("SIMD: AVX-2 (256-bit registers, fast)"),
        yallma3_llm::util::SimdLevel::Scalar => println!("SIMD: None (scalar fallback, slow)"),
    }

    let parallelism_config = ParallelismConfig::from_args(args.clone());
    if let Err(e) = configure_rayon(&parallelism_config) {
        eprintln!("Warning: Could not configure Rayon: {}", e);
        eprintln!("Continuing with default thread pool...");
    }
    print_parallelism_config(&parallelism_config);

    let model_path = args
        .iter()
        .position(|arg| arg == "--model_path")
        .and_then(|i| args.get(i + 1))
        .ok_or("Usage: inference --model_path <path> [--max_ctx_len <n>] [--threads <n>] [--parallelism <pct>] [--no-parallel] [--sequential]")?;

    let max_ctx_len = match args
        .iter()
        .position(|arg| arg == "--max_ctx_len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(0) => u32::MAX,
        Some(n) => n,
        None => 4096,
    };

    let model_type_override = args
        .iter()
        .position(|arg| arg == "--model-type")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| match s.to_lowercase().as_str() {
            "q8_0" => Some(WeightFormat::Q8_0),
            "q4_0" => Some(WeightFormat::Q4_0),
            "f16" => Some(WeightFormat::F16),
            _ => {
                eprintln!("Warning: unknown --model-type '{}', ignoring (valid: q8_0, q4_0, f16)", s);
                None
            }
        });

    let max_tokens = args
        .iter()
        .position(|arg| arg == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let transformer = load_transformer(model_path, parallelism_config.sequential, max_ctx_len, model_type_override)?;
    let mut sessions: Vec<InferenceSession> = Vec::new();
    let tokenizer = transformer.tokenizer().expect("Transformer did not load a tokenizer");

    let mut sampler = Sampler::new(transformer.vocab_size(), 0.7, 0.9, 42);

    println!("LLM Inference Console");
    println!("Model loaded from: {}", model_path);
    println!("Transformer type: {}", transformer.name());
    println!("Enter prompts (Ctrl+D or type '/exit' to exit):");

    let mut user_turn = true;
    let mut next: u32;
    let active_user: usize = 0;

    let system_prompt = "";
    let mut prompt_tokens: Vec<u32> = if tokenizer.chat_template.is_none() {
        tokenizer.apply_system_template(system_prompt)
    } else {
        if tokenizer.add_bos_token { vec![tokenizer.bos] } else { vec![] }
    };
    let mut user_prompt: String;

    loop {
        if user_turn {

            user_prompt = String::from("");
            print!("User: ");
            io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
            io::stdin()
                .read_line(&mut user_prompt)
                .expect("Failed to read line");
            if user_prompt.trim().is_empty() {
                continue;
            }
            if user_prompt.trim() == "/exit" || user_prompt.trim() == "/bye" {
                println!("Goodbye!");
                break;
            }

            prompt_tokens.extend(tokenizer.apply_turn_template(user_prompt.trim(), true));

            user_turn = false;
        } else {
            println!("\nyaLLMa3:");
            io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");

            #[cfg(feature = "debug_prints")]
            let show_n = prompt_tokens.len().min(100);
            #[cfg(feature = "debug_prints")]
            println!("prompt_tokens len={}, tokens={:?}", prompt_tokens.len(), &prompt_tokens[..show_n]);
            #[cfg(feature = "debug_prints")]
            println!("prompt_tokens text: {}", tokenizer.decode(&prompt_tokens[..show_n]));
            let session = get_or_create_session(&mut sessions, active_user, &*transformer);
            let prefill_pos = session.pos;
            let start = std::time::Instant::now();
            let mut logits = transformer.forward_x(&prompt_tokens, prefill_pos, session);
            session.pos += prompt_tokens.len() as u32;

            next = sampler.sample(&mut logits);

            let max_gen = if max_tokens > 0 { max_tokens } else { u32::MAX };
            let mut gen_count = 0u32;
            while next != transformer.eos_token()
                && session.pos < transformer.max_ctx_len()
                && gen_count < max_gen
            {
                let piece = tokenizer.decode(&[next]);
                print!("{}", piece);
                io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
                #[cfg(feature = "debug_prints")]
                println!(" [token_id={}]", next);
                io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");

                prompt_tokens.push(next);
                let gen_pos = session.pos;
                logits = transformer.forward_x(&[next], gen_pos, session);
                session.pos += 1;
                gen_count += 1;

                next = sampler.sample(&mut logits);
            }

            if gen_count >= max_gen && max_tokens > 0 {
                print!(" [max_tokens]");
                io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
            }

            let elapsed = start.elapsed();
            println!(" [{:.2?}]", elapsed);

            prompt_tokens.clear();
            if active_user < sessions.len() - 1 {
                sessions[active_user].reset();
            }
            user_turn = true;
        }
    }

    Ok(())
}

fn print_usage() {
    eprintln!("Usage: yallma3-infer --model_path <path> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model_path <path>   Path to model directory (required)");
    eprintln!("  --max_ctx_len <n>     Maximum context length (default: 4096, 0 = model max)");
    eprintln!("  --max-tokens <n>      Max tokens to generate (default: 0 = unlimited)");
    eprintln!("  --model-type <type>   Override weight format: q8_0, q4_0, or f16 (default: auto-detect)");
    eprintln!("  --threads <n>         Use exactly n threads for parallel operations");
    eprintln!("  --parallelism <pct>    Use percentage of available cores (0-100)");
    eprintln!("  --no-parallel         Disable parallelism (single-threaded)");
    eprintln!("  --sequential         Use sequential (non-optimized) attention");
}
