// Minimal Bun test for yallma3-llm FFI
// Note: Requires bun with FFI support

import { dlopen, FFIType } from "bun:ffi";

// Helper to encode string to buffer
function toCString(str) {
  return new TextEncoder().encode(str + "\0");
}

// Load the dynamic library
const lib = dlopen("target_llm/debug/libyallma3_llm.so", {
  load_transformer_ffi: {
    args: [FFIType.cstring],
    returns: FFIType.bool,
  },
  infer_ffi: {
    args: [FFIType.cstring],
    returns: FFIType.cstring,
  },
  free_string: {
    args: [FFIType.cstring],
    returns: FFIType.void,
  },
});

// Example usage
const modelPath =
  "/home/assem/work/PoC/_AI/lm.rs/models/llama/llama3.2-1b-it-q80.lmrs"; // Replace with actual path
const success = lib.symbols.load_transformer_ffi(toCString(modelPath));
if (success) {
  console.log("Transformer loaded successfully");
  const prompt = "Hello, world!";
  const responsePtr = lib.symbols.infer_ffi(toCString(prompt));
  // Read the cstring from pointer
  const response = Bun.unsafe.cstring(responsePtr);
  console.log("Response:", response);
  lib.symbols.free_string(responsePtr);
} else {
  console.log("Failed to load transformer");
}
