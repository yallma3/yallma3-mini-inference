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


use crate::util::{slice_to_f32, slice_to_u32};
use minijinja::Environment;
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::sync::LazyLock;

#[derive(Debug, Clone)]
struct TokenIndex {
    text: String,
    id: u32,
}

static GPT2_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap()
});

#[derive(Debug)]
pub struct Tokenizer {
    vocab: Vec<String>,
    pub bos: u32,
    pub eos: u32,
    vocab_scores: Vec<f32>,
    sorted_vocab: Vec<TokenIndex>,
    byte_level: bool,
    byte_to_token: Vec<u32>,
    token_to_byte: Vec<Option<u8>>,
    char_to_byte: std::collections::HashMap<char, u8>,
    pub chat_template: Option<String>,
    jinja_env: Option<Environment<'static>>,
    pre: String,
    pub add_bos_token: bool,
    pub add_space_prefix: bool,
}

impl Tokenizer {
    pub const BEGIN_OF_TEXT: u32 = 128000;
    pub const START_HEADER: u32 = 128006;
    pub const END_HEADER: u32 = 128007;
    pub const EOT: u32 = 128009;
    pub const HEADER_SEP: u32 = 271;

    pub(crate) fn build_gpt2_byte_table() -> [char; 256] {
        let mut bs: Vec<u8> = Vec::with_capacity(256);
        let mut cs: Vec<u32> = Vec::with_capacity(256);
        for b in b'!'..=b'~' { bs.push(b); cs.push(b as u32); }
        for b in 0xA1..=0xAC { bs.push(b); cs.push(b as u32); }
        for b in 0xAE..=0xFF { bs.push(b); cs.push(b as u32); }
        let mut n = 0u32;
        for b in 0..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut table = ['\0'; 256];
        for (&b, &cp) in bs.iter().zip(cs.iter()) {
            table[b as usize] = char::from_u32(cp).unwrap();
        }
        table
    }

    pub fn from_gguf_parts(vocab: Vec<String>, merges: Vec<String>, bos_id: u32, eos_id: u32, pre: String, add_bos_token: bool, add_space_prefix: bool) -> Tokenizer {
        let total_vsize = vocab.len();
        let mut vocab_scores = vec![0.0f32; total_vsize];
        let vocab_index: std::collections::HashMap<&str, usize> = vocab.iter()
            .enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        for (i, merge) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let merged = format!("{}{}", parts[0], parts[1]);
                if let Some(&idx) = vocab_index.get(merged.as_str()) {
                    vocab_scores[idx] = (merges.len() - i) as f32;
                }
            }
        }

        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(total_vsize);
        for (i, text) in vocab.iter().enumerate() {
            sorted_vocab.push(TokenIndex { text: text.clone(), id: i as u32 });
        }
        sorted_vocab.sort_by(|a, b| a.text.cmp(&b.text));

        let byte_table = Self::build_gpt2_byte_table();
        let mut byte_to_token = vec![0u32; 256];
        let mut token_to_byte = vec![None; total_vsize.max(256)];
        let mut char_to_byte = std::collections::HashMap::new();
        for byte in 0..=255u8 {
            let c = byte_table[byte as usize];
            char_to_byte.insert(c, byte);
            let c_str = c.to_string();
            if let Ok(idx) = sorted_vocab.binary_search_by(|t| t.text.cmp(&c_str)) {
                let tid = sorted_vocab[idx].id as usize;
                byte_to_token[byte as usize] = tid as u32;
                if tid < token_to_byte.len() {
                    token_to_byte[tid] = Some(byte);
                }
            }
        }

        Tokenizer {
            vocab,
            bos: bos_id,
            eos: eos_id,
            vocab_scores,
            sorted_vocab,
            byte_level: true,
            byte_to_token,
            token_to_byte,
            char_to_byte,
            chat_template: None,
            jinja_env: None,
            pre,
            add_bos_token,
            add_space_prefix,
        }
    }

    pub fn from_json(path: &std::path::Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let root: Value = serde_json::from_str(&text)?;

        let model = &root["model"];
        let vocab_obj = model["vocab"].as_object().unwrap();
        let added = root["added_tokens"].as_array().unwrap();

        let max_id = vocab_obj.values()
            .map(|v| v.as_u64().unwrap() as usize)
            .chain(added.iter().map(|t| t["id"].as_u64().unwrap() as usize))
            .max().unwrap_or(0);
        let total_vsize = max_id + 1;

        let mut vocab = vec![String::new(); total_vsize];
        for (token_str, id_val) in vocab_obj.iter() {
            let id = id_val.as_u64().unwrap() as usize;
            vocab[id] = token_str.clone();
        }
        for t in added.iter() {
            let id = t["id"].as_u64().unwrap() as usize;
            let content = t["content"].as_str().unwrap();
            vocab[id] = content.to_string();
        }

        let merges = model["merges"].as_array().unwrap();
        let mut vocab_scores = vec![0.0f32; total_vsize];
        let vocab_index: std::collections::HashMap<&str, usize> = vocab.iter()
            .enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        for (i, merge) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge.as_str().unwrap().splitn(2, ' ').collect();
            if parts.len() == 2 {
                let merged = format!("{}{}", parts[0], parts[1]);
                if let Some(&idx) = vocab_index.get(merged.as_str()) {
                    vocab_scores[idx] = (merges.len() - i) as f32;
                }
            }
        }

        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(total_vsize);
        for (i, text) in vocab.iter().enumerate() {
            sorted_vocab.push(TokenIndex { text: text.clone(), id: i as u32 });
        }
        sorted_vocab.sort_by(|a, b| a.text.cmp(&b.text));

        let mut bos = 128000u32;
        let mut eos = 128009u32;
        for t in added.iter() {
            let content = t["content"].as_str().unwrap();
            let id = t["id"].as_u64().unwrap() as u32;
            if content == "<|begin_of_text|>" { bos = id; }
            if content == "<|end_of_text|>" || content == "<|eot_id|>" { eos = id; }
        }

        let byte_table = Self::build_gpt2_byte_table();
        let mut byte_to_token = vec![0u32; 256];
        let mut token_to_byte = vec![None; total_vsize.max(256)];
        let mut char_to_byte = std::collections::HashMap::new();
        for byte in 0..=255u8 {
            let c = byte_table[byte as usize];
            char_to_byte.insert(c, byte);
            let c_str = c.to_string();
            if let Ok(idx) = sorted_vocab.binary_search_by(|t| t.text.cmp(&c_str)) {
                let tid = sorted_vocab[idx].id as usize;
                byte_to_token[byte as usize] = tid as u32;
                if tid < token_to_byte.len() {
                    token_to_byte[tid] = Some(byte);
                }
            }
        }

        Ok(Tokenizer {
            vocab,
            bos,
            eos,
            vocab_scores,
            sorted_vocab,
            byte_level: true,
            byte_to_token,
            token_to_byte,
            char_to_byte,
            chat_template: None,
            jinja_env: None,
            pre: "default".into(),
            add_bos_token: true,
            add_space_prefix: false,
        })
    }

    pub fn new(path: &std::path::Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
        let data: Vec<u8> = fs::read(path)?;

        let vocab_size = slice_to_u32(&data[0..4]);
        let bos = slice_to_u32(&data[8..12]);
        let eos = slice_to_u32(&data[12..16]);

        let mut vocab: Vec<String> = vec![];
        let mut vocab_scores: Vec<f32> = vec![];
        let mut sorted_vocab: Vec<TokenIndex> = vec![];

        let mut offset: usize = 16;

        for _ in 0..vocab_size {
            let score = slice_to_f32(&data[offset..offset + 4]);
            vocab_scores.push(score);
            offset += 4;
            let str_len = slice_to_u32(&data[offset..offset + 4]);
            offset += 4;
            let token_str = String::from_utf8(data[offset..offset + str_len as usize].to_vec())?;
            vocab.push(token_str);
            offset += str_len as usize;
        }

        for i in 0..vocab_size as usize {
            sorted_vocab.push(TokenIndex {
                text: vocab[i].clone(),
                id: i as u32,
            });
        }
        sorted_vocab.sort_by(|a, b| a.text.cmp(&b.text));

        Ok(Tokenizer {
            vocab,
            bos,
            eos,
            vocab_scores,
            sorted_vocab,
            byte_level: false,
            byte_to_token: (0..256).map(|b| b as u32 + 3).collect(),
            token_to_byte: vec![None; 0],
            char_to_byte: std::collections::HashMap::new(),
            chat_template: None,
            jinja_env: None,
            pre: "default".into(),
            add_bos_token: true,
            add_space_prefix: false,
        })
    }

    fn has_llama_tokens(&self) -> bool {
        self.vocab.len() > 128009
    }

    pub fn encode_wrap_n_header(
        &self,
        text: &str,
        header: &str,
        begin_of_text_wrap: bool,
    ) -> Vec<u32> {
        if !self.has_llama_tokens() {
            let mut tokens = Vec::new();
            if begin_of_text_wrap && self.add_bos_token {
                tokens.push(self.bos);
            }
            if ! text.is_empty() {
                tokens.extend(self.encode(text, false, false));
            }
            return tokens;
        }
        let mut tokens: Vec<u32> = Vec::new();
        if begin_of_text_wrap {
            tokens.push(Self::BEGIN_OF_TEXT)
        };
        if ! header.is_empty() {
            tokens.push(Self::START_HEADER);
            tokens.extend(self.encode(header, false, false));
            tokens.push(Self::END_HEADER);
            tokens.push(Self::HEADER_SEP);
        }
        if ! text.is_empty() {
            tokens.extend(self.encode(text, false, false));
        }
        if begin_of_text_wrap {
            tokens.push(Self::HEADER_SEP);    
            tokens.push(Self::EOT)
        };
        tokens
    }

    fn byte_encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len());
        for &byte in text.as_bytes() {
            tokens.push(self.byte_to_token[byte as usize]);
        }
        tokens
    }

    fn bpe_merge(&self, tokens: &mut Vec<u32>) {
        loop {
            let mut best_score = -1e10f32;
            let mut best_id = 0u32;
            let mut best_idx = -1i32;

            for idx in 0..tokens.len().wrapping_sub(1) {
                let new_t = self.vocab[tokens[idx] as usize].clone()
                    + &self.vocab[tokens[idx + 1] as usize];

                if let Ok(index) = self
                    .sorted_vocab
                    .binary_search_by(|token| token.text.cmp(&new_t))
                {
                    let temp_t = &self.sorted_vocab[index];
                    if self.vocab_scores[temp_t.id as usize] > best_score {
                        best_score = self.vocab_scores[temp_t.id as usize];
                        best_id = temp_t.id;
                        best_idx = idx as i32;
                    }
                }
            }

            if best_idx == -1 {
                break;
            }

            tokens[best_idx as usize] = best_id;
            tokens.remove((best_idx + 1) as usize);
        }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        assert!(!text.is_empty(), "Text to encode should not be empty");

        let mut tokens: Vec<u32> = Vec::new();

        if bos {
            tokens.push(self.bos);
        }

        match self.pre.as_str() {
            "gpt2" | "smollm" => {
                let text = if self.add_space_prefix && !text.starts_with(' ') {
                    let mut s = String::with_capacity(text.len() + 1);
                    s.push(' ');
                    s.push_str(text);
                    s
                } else {
                    text.to_string()
                };
                for segment in GPT2_REGEX.find_iter(&text) {
                    let mut seg_tokens = self.byte_encode(segment.as_str());
                    self.bpe_merge(&mut seg_tokens);
                    tokens.extend(seg_tokens);
                }
            }
            _ => {
                if self.byte_level {
                    tokens.extend(self.byte_encode(text));
                } else {
                    for c in text.chars() {
                        let c_str = c.to_string();
                        match self
                            .sorted_vocab
                            .binary_search_by(|token| token.text.cmp(&c_str))
                        {
                            Ok(index) => tokens.push(self.sorted_vocab[index].id),
                            Err(_) => {
                                for b in c_str.into_bytes().iter() {
                                    tokens.push(*b as u32 + 3);
                                }
                            }
                        }
                    }
                }
                self.bpe_merge(&mut tokens);
            }
        }

        if eos {
            tokens.push(self.eos);
        }

        tokens
    }

    pub fn set_chat_template(&mut self, template: String) {
        #[cfg(feature = "debug_prints")]
        println!("Chat template: {}", template);

        let mut env = Environment::new();
        env.add_template_owned("chat", template.clone()).unwrap();
        self.jinja_env = Some(env);
        self.chat_template = Some(template);
    }

    pub fn render_chat_template(&self, messages: &[(&str, &str)], add_generation_prompt: bool) -> Vec<u32> {
        match &self.jinja_env {
            Some(env) => {
                let tmpl = env.get_template("chat").unwrap();

                let msgs: Vec<serde_json::Value> = messages.iter().map(|(role, content)| {
                    serde_json::json!({"role": role, "content": content})
                }).collect();
                let ctx = serde_json::json!({"messages": msgs, "add_generation_prompt": add_generation_prompt});

                let rendered = tmpl.render(&ctx).unwrap();
                if rendered.is_empty() {
                    return Vec::new();
                }
                self.encode_template_output(&rendered)
            }
            None => Vec::new(),
        }
    }

    fn encode_template_output(&self, text: &str) -> Vec<u32> {
        let mut text_to_id: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
        for (tid, t_text) in self.vocab.iter().enumerate() {
            if t_text.contains('<') {
                text_to_id.insert(t_text.as_str(), tid as u32);
            }
        }

        let bytes = text.as_bytes();
        let mut result = Vec::new();
        let mut pos = 0;

        while pos < bytes.len() {
            if bytes[pos] == b'<' {
                let mut best_id = None;
                let mut best_len = 0;
                for (&t_text, &tid) in &text_to_id {
                    let t_bytes = t_text.as_bytes();
                    if bytes[pos..].starts_with(t_bytes) && t_bytes.len() > best_len {
                        best_id = Some(tid);
                        best_len = t_bytes.len();
                    }
                }
                if let Some(tid) = best_id {
                    result.push(tid);
                    pos += best_len;
                    continue;
                }
            }
            let start = pos;
            pos += 1;
            while pos < bytes.len() && bytes[pos] != b'<' {
                pos += 1;
            }
            result.extend(self.encode(&text[start..pos], false, false));
        }
        result
    }

    pub fn apply_system_template(&self, system: &str) -> Vec<u32> {
        match &self.chat_template {
            Some(_) => {
                let msgs = vec![("system", system)];
                self.render_chat_template(&msgs, false)
            }
            None => {
                if self.has_llama_tokens() {
                    self.encode_wrap_n_header(system, "system", true)
                } else {
                    let mut tokens = Vec::new();
                    if !system.is_empty() {
                        tokens.extend(self.encode(system, false, false));
                    }
                    tokens
                }
            }
        }
    }

    pub fn apply_turn_template(&self, user: &str, add_empty_assistant: bool) -> Vec<u32> {
        match &self.chat_template {
            Some(_) => {
                let mut msgs = Vec::new();
                msgs.push(("user", user));
                if add_empty_assistant {
                    msgs.push(("assistant", ""));
                }
                let mut result = self.render_chat_template(&msgs, add_empty_assistant);
                if add_empty_assistant && result.last() == Some(&Self::EOT) {
                    result.pop();
                }
                result
            }
            None => {
                let mut tokens = Vec::new();
                if self.has_llama_tokens() {
                    let user_id = self.encode("user", false, false);
                    let assistant_id = self.encode("assistant", false, false);
                    assert_eq!(user_id.len(), 1, "expected 'user' to be a single token");
                    assert_eq!(assistant_id.len(), 1, "expected 'assistant' to be a single token");
                    tokens.push(Self::START_HEADER);
                    tokens.push(user_id[0]);
                    tokens.push(Self::END_HEADER);
                    tokens.push(Self::HEADER_SEP);
                    tokens.extend(self.encode_wrap_n_header(user, "", false));
                    tokens.push(Self::EOT);
                    if add_empty_assistant {
                        tokens.push(Self::START_HEADER);
                        tokens.push(assistant_id[0]);
                        tokens.push(Self::END_HEADER);
                        tokens.push(Self::HEADER_SEP);
                    }
                } else {
                    if self.add_bos_token {
                        tokens.push(self.bos);
                    }
                    tokens.extend(self.encode(user, false, false));
                }
                tokens
            }
        }
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let vocab_len = self.vocab.len() as u32;
        if self.byte_level {
            let mut bytes: Vec<u8> = Vec::new();
            for &tid in token_ids {
                if let Some(Some(&b)) = self.token_to_byte.get(tid as usize).map(|ob| ob.as_ref()) {
                    bytes.push(b);
                } else if tid < vocab_len {
                    for c in self.vocab[tid as usize].chars() {
                        if let Some(&b) = self.char_to_byte.get(&c) {
                            bytes.push(b);
                        } else {
                            bytes.extend_from_slice(c.to_string().as_bytes());
                        }
                    }
                }
            }
            String::from_utf8_lossy(&bytes).to_string()
        } else {
            let mut text = String::new();
            for token_id in token_ids {
                if *token_id < vocab_len {
                    text.push_str(&self.vocab[*token_id as usize]);
                }
            }
            text
        }
    }
}
