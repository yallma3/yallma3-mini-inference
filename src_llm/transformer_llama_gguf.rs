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


use crate::base_transformer::{
    InferenceSession, ModelFamily, ModelWeights, TransformerBase, TransformerShape,
};
use crate::quantization::{MutableQuantizedTensorQ8, QuantType};
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use crate::weight_type::{MutableQuantizedTensorQ4_0, QuantizedTensor, WeightFormat};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::path::Path;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q8_1: u32 = 9;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_Q8_K: u32 = 15;


#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufValueType {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum GgufValue {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16),
    Uint32(u32), Int32(i32), Float32(f32), Bool(bool),
    String(String), Array(Vec<GgufValue>),
    Uint64(u64), Int64(i64), Float64(f64),
}

fn read_gguf_string(mmap: &[u8], offset: &mut usize) -> String {
    let len = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    let s = String::from_utf8(mmap[*offset..*offset + len].to_vec()).unwrap();
    *offset += len;
    s
}

fn read_gguf_value(mmap: &[u8], offset: &mut usize, value_type: u32) -> GgufValue {
    match value_type {
        0 => { let v = mmap[*offset]; *offset += 1; GgufValue::Uint8(v) }
        1 => { let v = mmap[*offset] as i8; *offset += 1; GgufValue::Int8(v) }
        2 => { let v = u16::from_le_bytes(mmap[*offset..*offset+2].try_into().unwrap()); *offset += 2; GgufValue::Uint16(v) }
        3 => { let v = i16::from_le_bytes(mmap[*offset..*offset+2].try_into().unwrap()); *offset += 2; GgufValue::Int16(v) }
        4 => { let v = u32::from_le_bytes(mmap[*offset..*offset+4].try_into().unwrap()); *offset += 4; GgufValue::Uint32(v) }
        5 => { let v = i32::from_le_bytes(mmap[*offset..*offset+4].try_into().unwrap()); *offset += 4; GgufValue::Int32(v) }
        6 => { let v = f32::from_le_bytes(mmap[*offset..*offset+4].try_into().unwrap()); *offset += 4; GgufValue::Float32(v) }
        7 => { let v = mmap[*offset] != 0; *offset += 1; GgufValue::Bool(v) }
        8 => { let s = read_gguf_string(mmap, offset); GgufValue::String(s) }
        9 => {
            let elem_type = u32::from_le_bytes(mmap[*offset..*offset+4].try_into().unwrap());
            *offset += 4;
            let arr_len = u64::from_le_bytes(mmap[*offset..*offset+8].try_into().unwrap());
            *offset += 8;
            let mut elements = Vec::with_capacity(arr_len as usize);
            for _ in 0..arr_len {
                elements.push(read_gguf_value(mmap, offset, elem_type));
            }
            GgufValue::Array(elements)
        }
        10 => { let v = u64::from_le_bytes(mmap[*offset..*offset+8].try_into().unwrap()); *offset += 8; GgufValue::Uint64(v) }
        11 => { let v = i64::from_le_bytes(mmap[*offset..*offset+8].try_into().unwrap()); *offset += 8; GgufValue::Int64(v) }
        12 => { let v = f64::from_le_bytes(mmap[*offset..*offset+8].try_into().unwrap()); *offset += 8; GgufValue::Float64(v) }
        _ => panic!("Unknown GGUF metadata value type: {}", value_type),
    }
}

pub struct TransformerLlamaGguf {
    base: TransformerBase,
    bos_id: u32,
    eos_id: u32,
    weight_format: WeightFormat,
}

fn get_metadata_string(meta: &HashMap<String, GgufValue>, key: &str) -> String {
    match meta.get(key) {
        Some(GgufValue::String(s)) => s.clone(),
        _ => panic!("Missing or wrong type for metadata key: {}", key),
    }
}

fn get_metadata_u64(meta: &HashMap<String, GgufValue>, key: &str) -> u64 {
    match meta.get(key) {
        Some(GgufValue::Uint64(v)) => *v,
        Some(GgufValue::Uint32(v)) => *v as u64,
        _ => panic!("Missing or wrong type for metadata key: {}", key),
    }
}

fn get_metadata_f32(meta: &HashMap<String, GgufValue>, key: &str) -> f32 {
    match meta.get(key) {
        Some(GgufValue::Float32(v)) => *v,
        Some(GgufValue::Float64(v)) => *v as f32,
        _ => panic!("Missing or wrong type for metadata key: {}", key),
    }
}

fn get_metadata_optional_u64(meta: &HashMap<String, GgufValue>, key: &str, default: u64) -> u64 {
    match meta.get(key) {
        Some(GgufValue::Uint64(v)) => *v,
        Some(GgufValue::Uint32(v)) => *v as u64,
        _ => default,
    }
}

fn get_metadata_optional_bool(meta: &HashMap<String, GgufValue>, key: &str, default: bool) -> bool {
    match meta.get(key) {
        Some(GgufValue::Bool(v)) => *v,
        Some(GgufValue::Uint8(v)) => *v != 0,
        Some(GgufValue::Uint32(v)) => *v != 0,
        Some(GgufValue::Uint64(v)) => *v != 0,
        _ => default,
    }
}

fn get_metadata_optional_string(meta: &HashMap<String, GgufValue>, key: &str, default: &str) -> String {
    match meta.get(key) {
        Some(GgufValue::String(s)) => s.clone(),
        _ => default.to_string(),
    }
}

fn get_metadata_optional_f32(meta: &HashMap<String, GgufValue>, key: &str, default: f32) -> f32 {
    match meta.get(key) {
        Some(GgufValue::Float32(v)) => *v,
        Some(GgufValue::Float64(v)) => {
            eprintln!("WARNING: {} is Float64, converting from {}", key, v);
            *v as f32
        },
        _ => {
            eprintln!("WARNING: {} not found or wrong type, defaulting to {}", key, default);
            default
        },
    }
}

impl TransformerLlamaGguf {
    pub fn load_model_with_override(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, fmt_override)
    }

    fn load_model_inner(model_path: &str, max_ctx_len: u32, fmt_override: Option<WeightFormat>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(model_path);
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        assert_eq!(magic, 0x46554747, "Not a GGUF file (bad magic)");

        let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        assert_eq!(_version, 3, "Only GGUF version 3 is supported, got version {}", _version);
        #[cfg(feature = "debug_prints")]
        println!("GGUF version: {}", _version);

        let tensor_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
        #[cfg(feature = "debug_prints")]
        println!("Tensors: {}, KV pairs: {}", tensor_count, kv_count);

        let mut offset: usize = 24;
        let mut metadata: HashMap<String, GgufValue> = HashMap::new();

        for _ in 0..kv_count {
            let key = read_gguf_string(&mmap, &mut offset);
            let value_type = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let value = read_gguf_value(&mmap, &mut offset, value_type);
            metadata.insert(key, value);
        }

        #[cfg(feature = "debug_prints")]
        println!("Parsed {} metadata KV pairs", metadata.len());

        let arch = get_metadata_string(&metadata, "general.architecture");
        assert_eq!(arch, "llama", "Architecture must be 'llama'");

        let dim = get_metadata_u64(&metadata, "llama.embedding_length") as u32;
        let hidden_dim = get_metadata_u64(&metadata, "llama.feed_forward_length") as u32;
        let n_layers = get_metadata_u64(&metadata, "llama.block_count") as u32;
        let n_heads = get_metadata_u64(&metadata, "llama.attention.head_count") as u32;
        let n_kv_heads = get_metadata_optional_u64(&metadata, "llama.attention.head_count_kv", n_heads as u64) as u32;
        let rms_norm_eps = get_metadata_f32(&metadata, "llama.attention.layer_norm_rms_epsilon");
        let file_type = get_metadata_optional_u64(&metadata, "general.file_type", 7);
        #[cfg(feature = "debug_prints")]
        println!("general.file_type={}", file_type);

        let rope_theta = get_metadata_optional_f32(&metadata, "llama.rope.freq_base", 10000.0);
        let seq_len = get_metadata_u64(&metadata, "llama.context_length") as u32;

        let head_size = dim / n_heads;
        assert_eq!(dim % n_heads, 0, "dim ({}) must be divisible by n_heads ({})", dim, n_heads);
        let ctx_len = seq_len.min(max_ctx_len);

        let vocab_size = match &metadata.get("tokenizer.ggml.tokens")
            .expect("tokenizer.ggml.tokens metadata key not found")
        {
            GgufValue::Array(arr) => arr.len() as u32,
            _ => panic!("tokenizer.ggml.tokens must be an array of strings"),
        };

        let rope_freq_scale = {
            let raw: f32 = get_metadata_optional_f32(&metadata, "llama.rope.scaling.factor", 1.0);
            if raw != 0.0 { 1.0 / raw } else { 1.0 }
        };
        let rope_ext_factor = get_metadata_optional_f32(&metadata, "llama.rope.scaling.ext_factor", 0.0);
        let rope_attn_factor = get_metadata_optional_f32(&metadata, "llama.rope.scaling.attn_factor", 1.0);
        let rope_original_ctx_len = get_metadata_optional_u64(&metadata, "llama.rope.scaling.original_context_length", seq_len as u64) as u32;
        let attention_scale = get_metadata_optional_f32(&metadata, "llama.attention.scale", 0.0);

        let (q_type, weight_format) = match fmt_override {
            Some(f) => match f {
                WeightFormat::Q8_0 => (QuantType::Q8_0, WeightFormat::Q8_0),
                WeightFormat::Q4_0 => (QuantType::Q4_0, WeightFormat::Q4_0),
                WeightFormat::F16 => (QuantType::F16, WeightFormat::F16),
            },
            None => match file_type {
                1 => (QuantType::F16, WeightFormat::F16),
                2 => (QuantType::Q4_0, WeightFormat::Q4_0),
                7 => (QuantType::Q8_0, WeightFormat::Q8_0),
                _ => {
                    let name = ggml_type_name(file_type as u32);
                    panic!("Unsupported GGUF file_type: {} ({}) — supported: F16=1, Q4_0=2, Q8_0=7",
                        file_type, name);
                }
            },
        };

        let shape = TransformerShape {
            dimension: dim,
            hidden_dimension: hidden_dim,
            n_heads,
            head_size,
            n_kv_heads,
            vocab_size,
            n_layers,
            ctx_len,
            rms_norm_eps,
            rope_theta,
            q_type,
            group_size: 32,
            rope_freq_scale,
            rope_ext_factor,
            rope_attn_factor,
            rope_original_ctx_len,
            attention_scale,
        };

        struct TensorInfo {
            name: String,
            #[allow(dead_code)]
            dims: Vec<u64>,
            tensor_type: u32,
            data_offset: u64,
            n_elements: usize,
        }

        let mut tensor_infos: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name = read_gguf_string(&mmap, &mut offset);
            let n_dims = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let mut dims = Vec::with_capacity(n_dims as usize);
            let mut n_elements = 1usize;
            for _ in 0..n_dims {
                let d = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
                offset += 8;
                dims.push(d);
                n_elements *= d as usize;
            }
            let tensor_type = u32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let data_offset = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
            offset += 8;

            tensor_infos.push(TensorInfo { name, dims, tensor_type, data_offset, n_elements });
        }

        let alignment: u64 = match metadata.get("general.alignment") {
            Some(GgufValue::Uint64(v)) => *v,
            Some(GgufValue::Uint32(v)) => *v as u64,
            _ => 32,
        };
        assert!(alignment.is_power_of_two(), "GGUF alignment must be a power of 2, got {}", alignment);
        #[cfg(feature = "debug_prints")]
        println!("Tensor data alignment: {}", alignment);
        let tensor_data_start = (offset as u64 + alignment - 1) / alignment * alignment;
        #[cfg(feature = "debug_prints")]
        println!("Tensor data section starts at byte {}", tensor_data_start);

        let tensor_map: HashMap<&str, &TensorInfo> = tensor_infos.iter().map(|ti| (ti.name.as_str(), ti)).collect();

        fn load_f32_tensor(mmap: &[u8], data_start: u64, info: &TensorInfo) -> Vec<f32> {
            assert_eq!(info.tensor_type, GGML_TYPE_F32, "Expected F32 tensor");
            let start = (data_start + info.data_offset) as usize;
            let n = info.n_elements;
            let mut result = vec![0.0f32; n];
            for i in 0..n {
                result[i] = f32::from_le_bytes(mmap[start + i * 4..][..4].try_into().unwrap());
            }
            result
        }

        fn load_q8_0_tensor(mmap: &[u8], data_start: u64, info: &TensorInfo) -> QuantizedTensor {
            let start = (data_start + info.data_offset) as usize;
            let n = info.n_elements;
            let blocks = (n + 31) / 32;
            let block_size_bytes: usize = 34;

            let mut qt = MutableQuantizedTensorQ8 {
                quant_vals: vec![0i8; n],
                scale_factor: vec![0.0f32; blocks],
            };

            for b in 0..blocks {
                let block_start = start + b * block_size_bytes;

                let scale_bits = u16::from_le_bytes(mmap[block_start..block_start + 2].try_into().unwrap());
                let scale = half::f16::from_bits(scale_bits).to_f32();
                qt.scale_factor[b] = scale;

                let qs_start = block_start + 2;
                for j in 0..32 {
                    let idx = b * 32 + j;
                    if idx < n {
                        qt.quant_vals[idx] = mmap[qs_start + j] as i8;
                    }
                }
            }

            QuantizedTensor::Q8(qt)
        }

        fn load_q4_0_tensor(mmap: &[u8], data_start: u64, info: &TensorInfo) -> QuantizedTensor {
            let start = (data_start + info.data_offset) as usize;
            let n = info.n_elements;
            let blocks = (n + 31) / 32;
            let block_size_bytes: usize = 18;

            let mut qt = MutableQuantizedTensorQ4_0 {
                quant_vals: vec![0i8; n],
                scale_factor: vec![0.0f32; blocks],
            };

            for b in 0..blocks {
                let block_start = start + b * block_size_bytes;

                let scale_bits = u16::from_le_bytes(mmap[block_start..block_start + 2].try_into().unwrap());
                let scale = half::f16::from_bits(scale_bits).to_f32();
                qt.scale_factor[b] = scale;

                let qs_start = block_start + 2;
                for j in 0..16 {
                    let byte = mmap[qs_start + j];
                    let lo = (byte & 0x0F) as i8;
                    let hi = ((byte >> 4) & 0x0F) as i8;
                    let idx_lo = b * 32 + j * 2;
                    let idx_hi = b * 32 + j * 2 + 1;
                    if idx_lo < n {
                        qt.quant_vals[idx_lo] = lo;
                    }
                    if idx_hi < n {
                        qt.quant_vals[idx_hi] = hi;
                    }
                }
            }

            QuantizedTensor::Q4_0(qt)
        }

        fn load_f16_tensor(mmap: &[u8], data_start: u64, info: &TensorInfo) -> QuantizedTensor {
            let start = (data_start + info.data_offset) as usize;
            let n = info.n_elements;
            let mut result = vec![0.0f32; n];
            for i in 0..n {
                let bits = u16::from_le_bytes(mmap[start + i * 2..][..2].try_into().unwrap());
                result[i] = half::f16::from_bits(bits).to_f32();
            }
            QuantizedTensor::F16(result)
        }

        fn ggml_type_name(t: u32) -> &'static str {
            match t {
                GGML_TYPE_F32 => "F32",
                GGML_TYPE_F16 => "F16",
                GGML_TYPE_Q4_0 => "Q4_0",
                GGML_TYPE_Q4_1 => "Q4_1",
                GGML_TYPE_Q5_0 => "Q5_0",
                GGML_TYPE_Q5_1 => "Q5_1",
                GGML_TYPE_Q8_0 => "Q8_0",
                GGML_TYPE_Q8_1 => "Q8_1",
                GGML_TYPE_Q2_K => "Q2_K",
                GGML_TYPE_Q3_K => "Q3_K",
                GGML_TYPE_Q4_K => "Q4_K",
                GGML_TYPE_Q5_K => "Q5_K",
                GGML_TYPE_Q6_K => "Q6_K",
                GGML_TYPE_Q8_K => "Q8_K",
                _ => "unknown",
            }
        }

        fn load_gguf_quantized_tensor(mmap: &[u8], data_start: u64, info: &TensorInfo) -> QuantizedTensor {
            match info.tensor_type {
                GGML_TYPE_Q8_0 => load_q8_0_tensor(mmap, data_start, info),
                GGML_TYPE_Q4_0 => load_q4_0_tensor(mmap, data_start, info),
                GGML_TYPE_F16 => load_f16_tensor(mmap, data_start, info),
                _ => panic!("Unsupported tensor type {} ({}) for {} — only Q8_0, Q4_0, and F16 are supported",
                    info.tensor_type, ggml_type_name(info.tensor_type), info.name),
            }
        }

        // Token embedding: load quantized, then dequantize to f32 in parallel
        let token_embedding_quant = {
            let ti = tensor_map.get("token_embd.weight").expect("token_embd.weight not found");
            Some(load_gguf_quantized_tensor(&mmap, tensor_data_start, ti))
        };
        let gs_gguf = shape.group_size as usize;
        let token_embedding = match token_embedding_quant.as_ref().unwrap() {
            QuantizedTensor::Q8(ref q) => {
                let dequant_len = q.quant_vals.len();
                let mut emb = vec![0.0f32; dequant_len];
                let qv = &q.quant_vals;
                let sf = &q.scale_factor;
                emb.par_chunks_mut(gs_gguf).enumerate().for_each(|(g, chunk)| {
                    let scale = sf[g];
                    let base = g * gs_gguf;
                    for (j, val) in chunk.iter_mut().enumerate() {
                        *val = qv[base + j] as f32 * scale;
                    }
                });
                emb
            }
            QuantizedTensor::Q4_0(ref q) => {
                let dequant_len = q.quant_vals.len();
                let mut emb = vec![0.0f32; dequant_len];
                let qv = &q.quant_vals;
                let sf = &q.scale_factor;
                emb.par_chunks_mut(gs_gguf).enumerate().for_each(|(g, chunk)| {
                    let scale = sf[g];
                    let base = g * gs_gguf;
                    for (j, val) in chunk.iter_mut().enumerate() {
                        *val = qv[base + j] as f32 * scale;
                    }
                });
                emb
            }
            QuantizedTensor::F16(ref v) => v.clone(),
        };

        // Final norm
        let w_rms_final = {
            let ti = tensor_map.get("output_norm.weight").expect("output_norm.weight not found");
            load_f32_tensor(&mmap, tensor_data_start, ti)
        };

        // Parallel: load all layer norms (F32)
        let n_layers_usize = n_layers as usize;
        let norm_names: Vec<(String, String)> = (0..n_layers_usize)
            .map(|l| {
                (format!("blk.{}.attn_norm.weight", l),
                 format!("blk.{}.ffn_norm.weight", l))
            })
            .collect();
        let norm_results: Vec<(Vec<f32>, Vec<f32>)> = norm_names.par_iter()
            .map(|(att_name, ffn_name)| {
                let att = load_f32_tensor(&mmap, tensor_data_start,
                    tensor_map.get(att_name.as_str()).unwrap_or_else(|| panic!("Missing tensor: {}", att_name)));
                let ffn = load_f32_tensor(&mmap, tensor_data_start,
                    tensor_map.get(ffn_name.as_str()).unwrap_or_else(|| panic!("Missing tensor: {}", ffn_name)));
                (att, ffn)
            })
            .collect();
        let mut w_rms_att = Vec::with_capacity(n_layers_usize);
        let mut w_rms_post_att = Vec::with_capacity(n_layers_usize);
        for (att, ffn) in norm_results {
            w_rms_att.push(att);
            w_rms_post_att.push(ffn);
        }

        // Parallel: load all per-layer quantized tensors
        struct LayerWeights {
            wq: QuantizedTensor,
            wk: QuantizedTensor,
            wv: QuantizedTensor,
            wo: QuantizedTensor,
            w1: QuantizedTensor,
            w2: QuantizedTensor,
            w3: QuantizedTensor,
        }

        let layer_tensors: Vec<LayerWeights> = (0..n_layers_usize).into_par_iter().map(|layer| {
            let prefix = format!("blk.{}", layer);

            let load_q = |proj: &str| {
                let tensor_name = format!("{}.{}.weight", prefix, proj);
                let ti = tensor_map.get(tensor_name.as_str())
                    .unwrap_or_else(|| panic!("Missing tensor: {}", tensor_name));
                load_gguf_quantized_tensor(&mmap, tensor_data_start, ti)
            };

            LayerWeights {
                wq: load_q("attn_q"),
                wk: load_q("attn_k"),
                wv: load_q("attn_v"),
                wo: load_q("attn_output"),
                w1: load_q("ffn_gate"),
                w2: load_q("ffn_down"),
                w3: load_q("ffn_up"),
            }
        }).collect();

        let mut wq = Vec::with_capacity(n_layers_usize);
        let mut wk = Vec::with_capacity(n_layers_usize);
        let mut wv = Vec::with_capacity(n_layers_usize);
        let mut wo = Vec::with_capacity(n_layers_usize);
        let mut w1 = Vec::with_capacity(n_layers_usize);
        let mut w2 = Vec::with_capacity(n_layers_usize);
        let mut w3 = Vec::with_capacity(n_layers_usize);
        for lt in layer_tensors {
            wq.push(lt.wq);
            wk.push(lt.wk);
            wv.push(lt.wv);
            wo.push(lt.wo);
            w1.push(lt.w1);
            w2.push(lt.w2);
            w3.push(lt.w3);
        }

        #[cfg(feature = "debug_prints")]
        println!("Weights loaded: {} layers, {} norms, {} ffn, rope_theta={}",
                 n_layers_usize, w_rms_att.len(), w1.len(), rope_theta);

        let model_weights = ModelWeights {
            token_embedding,
            token_embedding_quant,
            wq, wk, wv, wo,
            w_rms_att,
            w1, w2, w3,
            w_rms_post_att,
            w_rms_final,
        };

        let vocab = match metadata.get("tokenizer.ggml.tokens")
            .expect("tokenizer.ggml.tokens not found")
        {
            GgufValue::Array(arr) => {
                let mut v = Vec::with_capacity(arr.len());
                for elem in arr {
                    match elem {
                        GgufValue::String(s) => v.push(s.clone()),
                        _ => panic!("tokenizer.ggml.tokens element not a string"),
                    }
                }
                v
            }
            _ => panic!("tokenizer.ggml.tokens not an array"),
        };

        let merges = match metadata.get("tokenizer.ggml.merges") {
            Some(GgufValue::Array(arr)) => {
                let mut m = Vec::with_capacity(arr.len());
                for elem in arr {
                    match elem {
                        GgufValue::String(s) => m.push(s.clone()),
                        _ => panic!("tokenizer.ggml.merges element not a string"),
                    }
                }
                m
            }
            _ => vec![],
        };

        let bos_id = match metadata.get("tokenizer.ggml.bos_token_id") {
            Some(GgufValue::Uint32(v)) => *v,
            Some(GgufValue::Int32(v)) => *v as u32,
            Some(GgufValue::Uint64(v)) => *v as u32,
            Some(GgufValue::Int64(v)) => *v as u32,
            _ => 128000u32,
        };

        let eos_id = match metadata.get("tokenizer.ggml.eos_token_id") {
            Some(GgufValue::Uint32(v)) => *v,
            Some(GgufValue::Int32(v)) => *v as u32,
            Some(GgufValue::Uint64(v)) => *v as u32,
            Some(GgufValue::Int64(v)) => *v as u32,
            _ => 128009u32,
        };

        let pre = get_metadata_optional_string(&metadata, "tokenizer.ggml.pre", "default");
        let add_bos_token = get_metadata_optional_bool(&metadata, "tokenizer.ggml.add_bos_token", true);
        let add_space_prefix = get_metadata_optional_bool(&metadata, "tokenizer.ggml.add_space_prefix", false);
        let mut tokenizer = Tokenizer::from_gguf_parts(vocab, merges, bos_id, eos_id, pre, add_bos_token, add_space_prefix);
        if let Some(GgufValue::String(tmpl)) = metadata.get("tokenizer.chat_template") {
            tokenizer.set_chat_template(tmpl.clone());
            #[cfg(feature = "debug_prints")]
            println!("Loaded chat_template from GGUF metadata ({} chars)", tmpl.len());
        } else {
            #[cfg(feature = "debug_prints")]
            println!("No chat_template in GGUF metadata, using default template");
        }

        Ok(Self {
            bos_id,
            eos_id,
            weight_format,
            base: TransformerBase {
                model_info: format!("Llama GGUF model loaded from {}", path.display()),
                model_family: Some(ModelFamily::Llama),
                shape: Some(shape),
                weights: Some(model_weights),
                tokenizer: Some(tokenizer),
            },
        })
    }
}

impl Transformer for TransformerLlamaGguf {
    fn load_model(model_path: &str, max_ctx_len: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_model_inner(model_path, max_ctx_len, None)
    }

    fn name(&self) -> &str {
        "TransformerLlamaGguf"
    }

    fn vocab_size(&self) -> usize {
        self.base
            .shape
            .as_ref()
            .map(|s| s.vocab_size as usize)
            .unwrap_or(0)
    }

    /// LLaMA forward pass for GGUF models — matches llama.cpp's graph algebra.
    ///
    /// Each layer: RMS norm → quantize → QKV matmuls → RoPE (rope_gguf) →
    /// scaled dot-product attention (fused_attention) → WO projection →
    /// residual add → RMS norm → quantize → FFN gate/up (SiLU) → FFN down →
    /// residual add. Ends with final RMS norm → lm_head logits.
    ///
    /// RoPE note: Uses rope_gguf (consecutive-element pairs) because the GGUF
    /// converter permuted Q/K weight dimensions. This produces the same numerical
    /// result as first-half-second-half RoPE on unpermuted HuggingFace weights:
    ///
    ///   HF layout → first-half-second-half RoPE
    ///   Permuted HF layout → consecutive-pair RoPE  (this function, same result)
    ///
    /// Attention scale is read from GGUF metadata; defaults to 1/sqrt(head_size).
    fn forward_x(&self, tokens: &[u32], start_pos: u32, session: &mut InferenceSession) -> Vec<f32> {
        let shape = self.base.shape.as_ref().expect("Shape not initialized");
        let weights = self.base.weights.as_ref().expect("Weights not initialized");
        crate::forward::llama_forward_x(
            self.weight_format,
            crate::util::rope_gguf,
            tokens, start_pos, session,
            shape, weights,
            &|t| self.get_embeddings_batch(t),
        )
    }

    fn create_session(&self) -> InferenceSession {
        InferenceSession::new(self.base.shape.as_ref().expect("Shape not initialized"))
    }

    fn get_embeddings(&self, token: u32) -> Vec<f32> {
        self.base.get_embeddings(token)
    }

    fn get_embeddings_batch(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.base.get_embeddings_batch(tokens)
    }

    fn max_ctx_len(&self) -> u32 {
        self.base.shape.as_ref().map(|s| s.ctx_len).unwrap_or(2048)
    }

    fn eos_token(&self) -> u32 {
        self.eos_id
    }

    fn bos_token(&self) -> u32 {
        self.bos_id
    }

    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.base.tokenizer.as_ref()
    }
}

impl TransformerLlamaGguf {
    pub fn to_string(&self) -> String {
        self.base.to_string()
    }
}



