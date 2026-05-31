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


//! Parallelism configuration for controlling Rayon thread pool.
//!
//! This module provides configurable parallelism to balance inference performance
//! with system responsiveness. Rayon uses all CPU cores by default, which can
//! make the system unresponsive during heavy inference.
//!
//! # CLI Arguments
//!
//! - `--threads <n>` - Use exactly n threads for parallel operations
//! - `--parallelism <pct>` - Use percentage of available cores (0-100)
//! - `--no-parallel` - Disable parallelism entirely (single-threaded)
//!
//! # Examples
//!
//! ```bash
//! # Use 4 threads
//! inference --threads 4 --model_path model.gguf
//!
//! # Use 50% of available cores
//! inference --parallelism 50 --model_path model.gguf
//!
//! # Single-threaded mode
//! inference --no-parallel --model_path model.gguf
//! ```
//!
//! # Default Behavior
//!
//! Without any arguments, Rayon uses 50% of available CPU cores.
//! This balances inference performance with system responsiveness.

use num_cpus;
use rayon::ThreadPoolBuilder;

/// Configuration for the parallelism settings.
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    /// Number of threads to use. `None` means use all available cores.
    pub num_threads: Option<usize>,
    /// Whether parallelism is disabled entirely.
    pub disabled: bool,
    /// Whether to use the sequential (non-optimized) attention implementation.
    pub sequential: bool,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        // Default to 50% of available CPU cores to balance performance
        // with system responsiveness
        let available = num_cpus::get();
        let half = ((available as f64) * 0.5).ceil() as usize;
        Self {
            num_threads: Some(half.max(1)),
            disabled: false,
            sequential: false,
        }
    }
}

impl ParallelismConfig {
    /// Parses CLI arguments to extract parallelism configuration.
    ///
    /// Supports the following arguments:
    /// - `--threads <n>` - Use exactly n threads
    /// - `--parallelism <pct>` - Use percentage of available cores (0-100)
    /// - `--no-parallel` - Disable parallelism entirely
    ///
    /// # Arguments
    ///
    /// * `args` - Command line arguments (typically `std::env::args()`)
    ///
    /// # Returns
    ///
    /// A `ParallelismConfig` with the parsed settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use yallma3_llm::ParallelismConfig;
    /// let args = vec!["prog".to_string(), "--threads".to_string(), "4".to_string()];
    /// let config = ParallelismConfig::from_args(args);
    /// println!("Threads: {:?}", config.num_threads);
    /// ```
    pub fn from_args<I>(args: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut config = Self::default();
        let args: Vec<String> = args.into_iter().collect();

        // Check for --no-parallel first (disables everything)
        if args.iter().any(|arg| arg == "--no-parallel") {
            config.disabled = true;
            config.num_threads = Some(1);
            return config;
        }

        // Parse --threads <n>
        if let Some(idx) = args.iter().position(|arg| arg == "--threads") {
            if let Some(num_str) = args.get(idx + 1) {
                if let Ok(num) = num_str.parse::<usize>() {
                    if num > 0 {
                        config.num_threads = Some(num);
                    }
                }
            }
        }

        // Parse --parallelism <pct>
        if let Some(idx) = args.iter().position(|arg| arg == "--parallelism") {
            if let Some(pct_str) = args.get(idx + 1) {
                if let Ok(pct) = pct_str.parse::<f64>() {
                    if pct > 0.0 && pct <= 100.0 {
                        let available_cpus = num_cpus::get();
                        let threads = (available_cpus as f64 * pct / 100.0).ceil() as usize;
                        config.num_threads = Some(threads.max(1));
                    }
                }
            }
        }

        // Parse --sequential flag
        if args.iter().any(|arg| arg == "--sequential") {
            config.sequential = true;
        }

        config
    }

    /// Returns the effective number of threads that will be used.
    ///
    /// If parallelism is disabled, returns 1.
    /// If no explicit thread count is set, returns the total CPU count.
    pub fn effective_threads(&self) -> usize {
        if self.disabled {
            return 1;
        }
        self.num_threads.unwrap_or(num_cpus::get())
    }
}

/// Configures the global Rayon thread pool based on the provided configuration.
///
/// This function must be called before any parallel operations are performed.
/// It configures the global thread pool that Rayon will use for all parallel iterators.
///
/// # Arguments
///
/// * `config` - The parallelism configuration to apply
///
/// # Returns
///
/// Returns `Ok(())` if configuration succeeded, or an error if Rayon
/// initialization failed (e.g., if pool is already initialized).
///
/// # Behavior
///
/// - `num_threads = None`: Rayon uses all available CPU cores
/// - `num_threads = Some(n)`: Rayon uses exactly n threads
/// - `disabled = true`: Single-threaded mode (no parallelism)
///
/// # Example
///
/// ```
/// use yallma3_llm::{configure_rayon, ParallelismConfig};
/// let config = ParallelismConfig::default();
/// let _ = configure_rayon(&config);
/// ```
pub fn configure_rayon(config: &ParallelismConfig) -> Result<(), rayon::ThreadPoolBuildError> {
    let num_threads = config.effective_threads();

    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    Ok(())
}

/// Prints the parallelism configuration to stdout.
///
/// Displays the effective number of threads and the configuration source.
///
/// # Arguments
///
/// * `config` - The parallelism configuration to display
pub fn print_parallelism_config(config: &ParallelismConfig) {
    let available_cpus = num_cpus::get();
    let effective_threads = config.effective_threads();

    println!("Parallelism Configuration:");
    println!("  Available CPUs: {}", available_cpus);

    if config.disabled {
        println!("  Mode: Disabled (single-threaded)");
        println!("  Threads: 1");
    } else if let Some(threads) = config.num_threads {
        let is_default = {
            let half = ((available_cpus as f64) * 0.5).ceil() as usize;
            threads == half.max(1)
        };
        let source = if threads == available_cpus {
            "(all available)"
        } else if is_default {
            "(default: 50%)"
        } else {
            "(explicit)"
        };
        println!("  Mode: Enabled {}", source);
        println!("  Threads: {}", effective_threads);
        if threads < available_cpus {
            println!(
                "  Note: {} CPU core(s) will be idle",
                available_cpus - threads
            );
        }
    } else {
        println!("  Mode: Enabled (default: 50%)");
        println!("  Threads: {}", effective_threads);
    }
    println!();
}
