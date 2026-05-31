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


//! Tests for parallelism configuration.
//!
//! Run with: `cargo test --test parallelism_tests`

use yallma3_llm::ParallelismConfig;

#[test]
fn test_default_config() {
    let config = ParallelismConfig::default();
    assert!(!config.disabled);
    assert!(config.num_threads.is_some());
    let expected = ((num_cpus::get() as f64) * 0.5).ceil() as usize;
    assert_eq!(config.num_threads.unwrap(), expected.max(1));
    assert!(!config.sequential);
}

#[test]
fn test_no_parallel() {
    let args = vec![
        "inference".to_string(),
        "--model_path".to_string(),
        "model.gguf".to_string(),
        "--no-parallel".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    assert!(config.disabled);
    assert_eq!(config.num_threads, Some(1));
}

#[test]
fn test_threads_argument() {
    let args = vec![
        "inference".to_string(),
        "--threads".to_string(),
        "4".to_string(),
        "--model_path".to_string(),
        "model.gguf".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    assert!(!config.disabled);
    assert_eq!(config.num_threads, Some(4));
}

#[test]
fn test_parallelism_argument() {
    let args = vec![
        "inference".to_string(),
        "--parallelism".to_string(),
        "50".to_string(),
        "--model_path".to_string(),
        "model.gguf".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    assert!(!config.disabled);
    let available = num_cpus::get();
    let expected = ((available as f64 * 0.5).ceil() as usize).max(1);
    assert_eq!(config.num_threads, Some(expected));
}

#[test]
fn test_no_parallel_takes_precedence() {
    let args = vec![
        "inference".to_string(),
        "--no-parallel".to_string(),
        "--threads".to_string(),
        "8".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    // --no-parallel should take precedence
    assert!(config.disabled);
    assert_eq!(config.num_threads, Some(1));
}

#[test]
fn test_sequential_flag() {
    let args = vec![
        "inference".to_string(),
        "--model_path".to_string(),
        "model.gguf".to_string(),
        "--sequential".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    assert!(config.sequential);
}

#[test]
fn test_sequential_with_threads() {
    let args = vec![
        "inference".to_string(),
        "--threads".to_string(),
        "4".to_string(),
        "--sequential".to_string(),
    ];
    let config = ParallelismConfig::from_args(args);

    assert!(config.sequential);
    assert_eq!(config.num_threads, Some(4));
}

#[test]
fn test_effective_threads_disabled() {
    let config = ParallelismConfig {
        num_threads: Some(8),
        disabled: true,
        sequential: false,
    };
    assert_eq!(config.effective_threads(), 1);
}

#[test]
fn test_effective_threads_explicit() {
    let config = ParallelismConfig {
        num_threads: Some(4),
        disabled: false,
        sequential: false,
    };
    assert_eq!(config.effective_threads(), 4);
}

#[test]
fn test_effective_threads_default() {
    let config = ParallelismConfig::default();
    let available = num_cpus::get();
    let expected = ((available as f64) * 0.5).ceil() as usize;
    assert_eq!(config.effective_threads(), expected.max(1));
}
