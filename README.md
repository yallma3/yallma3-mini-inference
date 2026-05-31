# yaLLMa3 — Yet Another LLaMA 3 Inference Engine

A high-performance, from-scratch LLaMA 3 inference engine written in **Rust**, with SIMD-accelerated matrix operations and multi-format model support. Runs 1B–8B parameter models on consumer hardware.

## Features

- **Performance**: SIMD-optimized (AVX-512/AVX-2) matrix ops with parallelized attention and feed-forward layers via [Rayon](https://github.com/rayon-rs/rayon)
- **Quantization**: Q8_0 (8-bit), Q4_0 (4-bit), and F16 weight formats — fit larger models in less memory
- **Multi-format**: Load models from GGUF, lm.rs (`.lmrs`), and Safetensors (`.safetensors`)
- **Interactive console**: Chat loop with Jinja2 chat template support via [minijinja](https://github.com/mitsuhiko/minijinja)
- **FFI interface**: C-ABI bindings for calling inference from other languages (Bun.js PoC included)
- **Plug-and-play**: Works with LLaMA-family models (Llama 3, 3.2, SmolLM2, and any GGUF with `architecture = "llama"`)

## Quick Start

```bash
# Build
cargo build --release

# Run interactive inference
./target_llm/release/yallma3-infer --model_path models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

## Usage

```text
Usage: yallma3-infer --model_path <path> [options]

Options:
  --model_path <path>   Path to model directory (required)
  --max_ctx_len <n>     Maximum context length (default: 4096, 0 = model max)
  --max-tokens <n>      Max tokens to generate (default: 0 = unlimited)
  --model-type <type>   Override weight format: q8_0, q4_0, or f16
  --threads <n>         Use exactly n threads for parallel operations
  --parallelism <pct>   Use percentage of available cores (0-100)
  --no-parallel         Disable parallelism (single-threaded)
  --sequential          Use sequential (non-optimized) attention
```

### Examples

```bash
# With parallelism control
yallma3-infer --model_path model.gguf --threads 4

# Single-threaded
yallma3-infer --model_path model.gguf --no-parallel

# Safetensors model
yallma3-infer --model_path models/llama_q8_st/model.safetensors

# Override weight format
yallma3-infer --model_path model.gguf --model-type q4_0

# Longer context
yallma3-infer --model_path model.gguf --max_ctx_len 8192 --max-tokens 200
```

## Build

```bash
cargo build --release
```

Produces:
- `target_llm/release/yallma3-infer` — standalone binary
- `target_llm/release/libyallma3_llm.so` — shared library (FFI)

### Debug build with hex dumps

```bash
cargo build --features debug_prints
```

## Run Tests

```bash
cargo test
```

## FFI (C-ABI)

Load the shared library from any language that supports C FFI. See [`PoC_Bun_load.js`](./PoC_Bun_load.js) for a Bun.js example:

```javascript
import { dlopen, FFIType } from "bun:ffi";

const lib = dlopen("target_llm/debug/libyallma3_llm.so", {
  load_transformer_ffi: { args: [FFIType.cstring], returns: FFIType.bool },
  infer_ffi:             { args: [FFIType.cstring], returns: FFIType.cstring },
  free_string:           { args: [FFIType.cstring], returns: FFIType.void },
});
```

## Supported Models

Any model that fits the LLaMA architecture can be used. Tested with:
- **Llama 3.2 1B Instruct** (GGUF Q8_0, Q4_0, F16)
- **SmolLM2 135M / 360M** (GGUF Q8_0)

Models are auto-detected by file extension (`.gguf`, `.lmrs`, `.safetensors`). For Safetensors, a HuggingFace-style `config.json` is required alongside the weight files.

## Project Structure

```
src_llm/
├── main.rs                          # CLI entry point
├── lib.rs                           # Library root + FFI bindings
├── transformer.rs                   # Transformer trait
├── base_transformer.rs              # Core data structures
├── forward.rs                       # Parallel forward pass (SIMD)
├── parallelism.rs                   # Rayon thread pool config
├── quantization.rs                  # Q8_0 quantization
├── sampler.rs                       # Token sampling (top-p, temp)
├── tokenizer.rs                     # BPE tokenizer + chat templates
├── transformer_llama_gguf.rs        # GGUF loader
├── transformer_llama_lmrs.rs        # lm.rs loader (parallel)
├── transformer_llama_lmrs_seq.rs    # lm.rs loader — educational reference (sequential, fully commented)
├── transformer_llama_st.rs          # Safetensors loader
├── util.rs                          # SIMD detection, RMS norm, RoPE
└── weight_type.rs                   # Weight formats + matmul
```

## License

[Mozilla Public License 2.0](./LICENSE)
