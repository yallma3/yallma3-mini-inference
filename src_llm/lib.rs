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


use std::convert::TryInto;
use std::ffi::CStr;
use std::fs::File;
use std::path::Path;

pub mod base_transformer;
pub mod forward;
pub mod parallelism;
pub mod quantization;
pub mod sampler;
pub mod tokenizer;
pub mod transformer;
pub mod transformer_llama_lmrs;
pub mod transformer_llama_st_seq;
pub mod transformer_llama_st;
pub mod transformer_llama_gguf;
pub mod util;
pub mod weight_type;

pub use weight_type::WeightFormat;
pub use base_transformer::{InferenceSession, ModelFamily, TransformerBase, TransformerShape};
pub use parallelism::{configure_rayon, print_parallelism_config, ParallelismConfig};
pub use quantization::quantize;
pub use quantization::MutableQuantizedTensorQ8;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
pub use transformer::Transformer;
pub use transformer_llama_lmrs::TransformerLlamaLmrs;
pub use transformer_llama_st_seq::TransformerLlamaStSeq;
pub use transformer_llama_st::TransformerLlamaSt;
pub use transformer_llama_gguf::TransformerLlamaGguf;
pub use util::rms_norm;

static mut CURRENT_TRANSFORMER: Option<Box<dyn Transformer>> = None;

fn skip_gguf_value(mmap: &[u8], mut offset: usize, vtype: u32) -> usize {
    match vtype {
        0 | 1 | 7 => offset += 1,
        2 | 3     => offset += 2,
        4 | 5 | 6 => offset += 4,
        8 => {
            let len = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8 + len;
        }
        9 => {
            let elem_type = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let arr_len = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            for _ in 0..arr_len {
                offset = skip_gguf_value(mmap, offset, elem_type);
            }
        }
        10 | 11 | 12 => offset += 8,
        _ => panic!("Unknown GGUF value type for skip: {}", vtype),
    }
    offset
}

fn read_gguf_architecture(model_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
    if magic != 0x46554747 {
        return Err("Not a GGUF file".into());
    }

    let kv_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
    let mut offset: usize = 24;

    for _ in 0..kv_count {
        let key_len = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let key = String::from_utf8(mmap[offset..offset + key_len].to_vec()).unwrap();
        offset += key_len;

        let vtype = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
        offset += 4;

        if key == "general.architecture" {
            let s_len = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
            let value = String::from_utf8(mmap[offset + 8..offset + 8 + s_len].to_vec()).unwrap();
            return Ok(value);
        }

        offset = skip_gguf_value(&mmap, offset, vtype);
    }

    Err("general.architecture not found".into())
}

pub fn load_transformer(
    model_path: &str,
    sequential: bool,
    max_ctx_len: u32,
    fmt_override: Option<WeightFormat>,
) -> Result<Box<dyn Transformer>, Box<dyn std::error::Error>> {
    match Path::new(model_path).extension().and_then(|s| s.to_str()) {
        Some("gguf") => {
            let arch = read_gguf_architecture(model_path)?;
            match arch.as_str() {
                "llama" => Ok(Box::new(TransformerLlamaGguf::load_model_with_override(model_path, max_ctx_len, fmt_override)?)),
                _ => Err(format!("Unsupported GGUF architecture: {}", arch).into()),
            }
        }
        Some("lmrs") => {
            Ok(Box::new(TransformerLlamaLmrs::load_model_with_override(model_path, max_ctx_len, fmt_override)?))
        }
        Some("safetensors") => {
            if !sequential {
                Ok(Box::new(TransformerLlamaSt::load_model_with_override(model_path, max_ctx_len, fmt_override)?))
            } else {
                Ok(Box::new(TransformerLlamaStSeq::load_model_with_override(model_path, max_ctx_len, fmt_override)?))
            }
        }
        _ => Err(format!("Unsupported model extension for path: {}", model_path).into()),
    }
}

/// FFI function to load transformer
#[no_mangle]
pub extern "C" fn load_transformer_ffi(model_path: *const std::os::raw::c_char) -> bool {
    if model_path.is_null() {
        return false;
    }

    let path_str = unsafe { CStr::from_ptr(model_path).to_str().unwrap_or("") };
    match load_transformer(path_str, false, u32::MAX, None) {
        Ok(transformer) => {
            unsafe {
                CURRENT_TRANSFORMER = Some(transformer);
            }
            true
        }
        Err(_) => false,
    }
}


