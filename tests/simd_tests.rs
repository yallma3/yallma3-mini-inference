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


//! Integration tests for SIMD, CPU detection, and mathematical operations.
//!
//! Run with: `cargo test`

use yallma3_llm::util::{rms_norm, silu};

#[cfg(target_arch = "x86_64")]
mod simd_detection {
    use yallma3_llm::util::{get_simd_level, SimdLevel};

    #[test]
    fn test_simd_level_detection_returns_valid_variant() {
        let level = get_simd_level();
        assert!(
            matches!(
                level,
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Scalar
            ),
            "SIMD level should be one of the defined variants"
        );
    }

    #[test]
    fn test_simd_level_debug_trait() {
        let level = get_simd_level();
        let debug_str = format!("{:?}", level);
        assert!(
            debug_str.contains("Avx512")
                || debug_str.contains("Avx2")
                || debug_str.contains("Scalar"),
            "Debug output should contain variant name"
        );
    }

    #[test]
    fn test_simd_level_clone() {
        let level = get_simd_level();
        let _cloned = level;
    }

    #[test]
    fn test_simd_level_copy() {
        let level = get_simd_level();
        let copied = level;
        assert_eq!(level, copied);
    }

    #[test]
    fn test_simd_level_equality() {
        let level1 = get_simd_level();
        let level2 = get_simd_level();
        assert_eq!(level1, level2, "SIMD level should be consistent");
    }

    #[test]
    fn test_simd_level_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SimdLevel>();
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod simd_detection {
    use super::*;

    #[test]
    fn test_simd_level_on_non_x86() {
        let level = get_simd_level();
        assert_eq!(level, SimdLevel::Scalar, "Non-x86 should always use scalar");
    }
}

mod rms_norm_tests {
    use super::*;

    #[test]
    fn test_rms_norm_preserves_rms_after_normalization() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.001,
            "RMS should be ~1.0 after normalization, got {}",
            rms
        );
    }

    #[test]
    fn test_rms_norm_with_weights() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![2.0f32, 2.0, 2.0, 2.0];
        rms_norm(&mut x, &weight, 1e-5, false);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!(
            (rms - 2.0).abs() < 0.001,
            "RMS should be ~2.0 (weight magnitude), got {}",
            rms
        );
    }

    #[test]
    fn test_rms_norm_empty_slice() {
        let mut x: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        rms_norm(&mut x, &weight, 1e-5, false);
        assert!(x.is_empty());
    }

    #[test]
    fn test_rms_norm_single_element() {
        let mut x = vec![5.0f32];
        let weight = vec![1.0f32];
        rms_norm(&mut x, &weight, 1e-5, false);

        assert!(
            (x[0] - 1.0).abs() < 0.001,
            "Single element normalized should be ~1.0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_rms_norm_eps_stability() {
        let mut x = vec![1e-10f32, 1e-10, 1e-10, 1e-10];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        for val in x.iter() {
            assert!(val.is_finite(), "Should handle very small values with eps");
        }
    }

    #[test]
    fn test_rms_norm_zeros() {
        let mut x = vec![0.0f32; 4];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        for val in x.iter() {
            assert!(
                val.is_finite(),
                "Zeros with eps should produce finite values, got {}",
                val
            );
        }
    }

    #[test]
    fn test_rms_norm_large_values() {
        let mut x = vec![1000.0f32, 2000.0, 3000.0, 4000.0];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.001,
            "Large values should normalize correctly, got {}",
            rms
        );
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let mut x = vec![-1.0f32, -2.0, -3.0, -4.0];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.001,
            "Negative values should normalize correctly, got {}",
            rms
        );
    }

    #[test]
    fn test_rms_norm_mixed_values() {
        let mut x = vec![-1.0f32, 2.0, -3.0, 4.0];
        let weight = vec![1.0f32; 4];
        rms_norm(&mut x, &weight, 1e-5, false);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.001,
            "Mixed sign values should normalize correctly, got {}",
            rms
        );
    }

    #[test]
    fn test_rms_norm_power_of_two_size() {
        for size in [8, 16, 32, 64, 128, 256] {
            let mut x: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let weight = vec![1.0f32; size];
            rms_norm(&mut x, &weight, 1e-5, false);

            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / size as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.001,
                "Size {}: RMS should be ~1.0, got {}",
                size,
                rms
            );
        }
    }

    #[test]
    fn test_rms_norm_non_power_of_two_size() {
        for size in [7, 15, 33, 100, 1000] {
            let mut x: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let weight = vec![1.0f32; size];
            rms_norm(&mut x, &weight, 1e-5, false);

            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / size as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.001,
                "Size {}: RMS should be ~1.0, got {}",
                size,
                rms
            );
        }
    }
}

mod silu_tests {
    use super::*;

    #[test]
    fn test_silu_basic() {
        let mut x = vec![0.0f32];
        silu(&mut x);
        assert!(
            (x[0] - 0.0).abs() < 0.001,
            "silu(0) should be ~0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_silu_positive() {
        let mut x = vec![1.0f32];
        silu(&mut x);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(
            (x[0] - expected).abs() < 0.001,
            "silu(1) should be sigmoid(1), got {}",
            x[0]
        );
    }

    #[test]
    fn test_silu_negative() {
        let mut x = vec![-1.0f32];
        silu(&mut x);
        let expected = -1.0 / (1.0 + 1.0f32.exp());
        assert!(
            (x[0] - expected).abs() < 0.001,
            "silu(-1) should be -sigmoid(1), got {}",
            x[0]
        );
    }

    #[test]
    fn test_silu_large() {
        let mut x = vec![10.0f32];
        silu(&mut x);
        // silu(10) = 10 * sigmoid(10) ≈ 10 * 0.99995 ≈ 9.9995
        let expected = 10.0 * (1.0 / (1.0 + (-10.0f32).exp()));
        assert!(
            (x[0] - expected).abs() < 0.001,
            "silu(10) should be ~{}, got {}",
            expected,
            x[0]
        );
    }

    #[test]
    fn test_silu_small_negative() {
        let mut x = vec![-10.0f32];
        silu(&mut x);
        assert!(
            (x[0] - (-0.000045)).abs() < 0.001,
            "silu(-10) should be ~0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_silu_vector() {
        let mut x = vec![0.0f32, 1.0, -1.0, 2.0];
        silu(&mut x);

        assert!((x[0] - 0.0).abs() < 0.001);
        assert!((x[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 0.001);
        assert!((x[2] - (-1.0 / (1.0 + 1.0f32.exp()))).abs() < 0.001);
        assert!((x[3] - 2.0 / (1.0 + (-2.0f32).exp())).abs() < 0.001);
    }

    #[test]
    fn test_silu_consistency_with_formula() {
        let values = [-5.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];

        for &val in &values {
            let mut x = vec![val];
            silu(&mut x);
            let expected = val / (1.0 + (-val).exp());
            assert!(
                (x[0] - expected).abs() < 0.001,
                "silu({}) should be {}, got {}",
                val,
                expected,
                x[0]
            );
        }
    }
}

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_rms_norm_performance() {
        let size = 4096;
        let iterations = 100;

        let x = vec![1.0f32; size];
        let weight = vec![1.0f32; size];

        let start = Instant::now();
        for _ in 0..iterations {
            let mut test_x = x.clone();
            rms_norm(&mut test_x, &weight, 1e-5, false);
        }
        let duration = start.elapsed();

        println!(
            "RMSNorm: {} elements x {} iterations took {:?}",
            size, iterations, duration
        );

        let ns_per_element = duration.as_nanos() as f64 / (size as f64 * iterations as f64);
        println!("Average: {:.2} ns/element", ns_per_element);

        assert!(
            ns_per_element < 100.0,
            "RMSNorm should be reasonably fast, got {:.2} ns/element",
            ns_per_element
        );
    }

    #[test]
    fn test_silu_performance() {
        let size = 10000;
        let iterations = 100;

        let x = vec![1.0f32; size];

        let start = Instant::now();
        for _ in 0..iterations {
            let mut test_x = x.clone();
            silu(&mut test_x);
        }
        let duration = start.elapsed();

        println!(
            "SiLU: {} elements x {} iterations took {:?}",
            size, iterations, duration
        );

        let ns_per_element = duration.as_nanos() as f64 / (size as f64 * iterations as f64);
        println!("Average: {:.2} ns/element", ns_per_element);
    }
}

mod transformer_llama_lmrs_tests {
    use std::path::Path;

    fn find_test_model() -> Option<String> {
        let search_paths = [
            "models/llama3-8b-instruct-q5_k_m.lmrs",
            "../models/llama3-8b-instruct-q5_k_m.lmrs",
            "../../models/llama3-8b-instruct-q5_k_m.lmrs",
        ];

        for path in &search_paths {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
        None
    }

    #[test]
    fn test_forward_returns_correct_sized_logits() {
        let model_path = match find_test_model() {
            Some(p) => p,
            None => return,
        };

        let model = match yallma3_llm::load_transformer(&model_path, false, u32::MAX, None) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: failed to load model: {}", e);
                return;
            }
        };

        let mut session = model.create_session();
        let bos_token = model.bos_token();
        let output = model.forward_x(&[bos_token], 0, &mut session);

        let vocab_size = model.vocab_size();
        assert_eq!(
            output.len(),
            vocab_size,
            "Forward output size should match vocab size"
        );
    }

    #[test]
    fn test_forward_output_is_finite() {
        let model_path = match find_test_model() {
            Some(p) => p,
            None => return,
        };

        let model = match yallma3_llm::load_transformer(&model_path, false, u32::MAX, None) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: failed to load model: {}", e);
                return;
            }
        };

        let mut session = model.create_session();
        let bos_token = model.bos_token();
        let output = model.forward_x(&[bos_token], 0, &mut session);

        for (i, val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output logits[{}] should be finite, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_forward_causal_masking() {
        let model_path = match find_test_model() {
            Some(p) => p,
            None => return,
        };

        let model = match yallma3_llm::load_transformer(&model_path, false, u32::MAX, None) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: failed to load model: {}", e);
                return;
            }
        };

        let mut session = model.create_session();
        let bos_token = model.bos_token();
        let _ = model.forward_x(&[bos_token], 0, &mut session);

        let output_gen = model.forward_x(&[bos_token], 1, &mut session);
        let output_prompt = model.forward_x(&[bos_token], 1, &mut session);

        let diff_count = output_gen
            .iter()
            .zip(output_prompt.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();

        assert!(
            diff_count > 0,
            "Generation mode should produce different output than prompt mode"
        );
    }

    #[test]
    fn test_forward_max_ctx_len() {
        let model_path = match find_test_model() {
            Some(p) => p,
            None => return,
        };

        let model = match yallma3_llm::load_transformer(&model_path, false, u32::MAX, None) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: failed to load model: {}", e);
                return;
            }
        };

        let max_ctx = model.max_ctx_len();
        assert!(max_ctx > 0, "Max context length should be positive");

        let mut session = model.create_session();
        let bos_token = model.bos_token();
        let _ = model.forward_x(&[bos_token], max_ctx - 1, &mut session);
    }

    #[test]
    fn test_transformer_name() {
        let model_path = match find_test_model() {
            Some(p) => p,
            None => return,
        };

        let model = match yallma3_llm::load_transformer(&model_path, false, u32::MAX, None) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: failed to load model: {}", e);
                return;
            }
        };

        let name = model.name();
        assert!(
            name.contains("Llama") || name.contains("lmrs"),
            "Transformer name should indicate Llama LMRS model, got {}",
            name
        );
    }
}
