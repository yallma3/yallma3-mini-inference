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


#[derive(Debug, Copy, Clone)]
struct ProbIndex {
    prob: f32,
    index: u32,
}

fn softmax(logits: &mut [f32]) {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits
        .iter_mut()
        .map(|x| {
            *x = (*x - max_logit).exp();
            *x
        })
        .sum();
    logits.iter_mut().for_each(|x| *x /= exp_sum);
}
/// Sample an index from a probability distribution
/// # Arguments
/// * `probs` - A slice of probabilities (should sum to 1)
/// # Returns
/// * `Option<u32>` - The sampled index, or None if input is invalid
/// # Example
/// let probs = [0.1, 0.5, 0.4];
/// let sampled_index = sample_from_probs(&probs);
/// println!("Sampled index: {:?}", sampled_index);

pub struct Sampler {
    vocab_size: usize,
    probability_idx: Vec<ProbIndex>,
    temperature: f32,
    top_p: f32,
    seed: u64,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler {
            vocab_size,
            probability_idx: vec![
                ProbIndex {
                    prob: 0.0,
                    index: 0
                };
                vocab_size as usize
            ],
            temperature,
            top_p,
            seed,
        }
    }

    fn sample_argmax(probabilities: &[f32]) -> u32 {
        let mut max_i: u32 = 0;
        let mut max_p = probabilities[0];

        for (i, p) in probabilities.iter().enumerate().skip(1) {
            if *p > max_p {
                max_i = i as u32;
                max_p = *p;
            }
        }

        max_i
    }
    //TODO: check the implementation
    fn random_u32(mut state: u64) -> u32 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;

        ((state.wrapping_mul(0x2545F4914F6CDD1Du64)) >> 32) as u32
    }

    fn random_f32(state: u64) -> f32 {
        (Self::random_u32(state) >> 8) as f32 / 16777216.0f32
    }

    fn sample_mult(probabilities: &[f32], rand: f32) -> u32 {
        let mut cdf: f32 = 0.0;
        let n = probabilities.len();

        for (i, p) in probabilities.iter().enumerate() {
            cdf += *p;
            if rand < cdf {
                return i as u32;
            }
        }

        (n - 1) as u32
    }

    fn compare(a: &ProbIndex, b: &ProbIndex) -> std::cmp::Ordering {
        if a.prob > b.prob {
            std::cmp::Ordering::Less
        } else if a.prob < b.prob {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    fn sample_topp(&mut self, probabilities: &[f32], top_p: f32, rand: f32) -> u32 {
        let n = probabilities.len();
        let mut n0 = 0;

        let cutoff: f32 = (1.0f32 - top_p) / (n - 1) as f32;

        for (i, p) in probabilities.iter().enumerate() {
            if *p >= cutoff {
                self.probability_idx[n0].index = i as u32;
                self.probability_idx[n0].prob = *p;
                n0 += 1;
            }
        }

        self.probability_idx.sort_by(Sampler::compare);

        let mut cumulative_prob: f32 = 0.0;

        let mut last_idx = n0 - 1;

        for i in 0..n0 {
            cumulative_prob += self.probability_idx[i].prob;
            if cumulative_prob > top_p {
                last_idx = i;
                break;
            }
        }

        let r = rand * cumulative_prob;
        let mut cdf: f32 = 0.0;

        for i in 0..last_idx + 1 {
            cdf += self.probability_idx[i].prob;
            if r < cdf {
                return self.probability_idx[i].index;
            }
        }

        self.probability_idx[last_idx].index
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let next: u32;

        if self.temperature == 0.0f32 {
            next = Sampler::sample_argmax(logits);
        } else {
            for q in 0..self.vocab_size {
                logits[q as usize] /= self.temperature;
            }

            softmax(logits);

            let rand: f32 = Self::random_f32(self.seed);

            if self.top_p <= 0.0 || self.top_p >= 1.0 {
                next = Sampler::sample_mult(logits, rand);
            } else {
                next = self.sample_topp(logits, self.top_p, rand);
            }
        }

        next
    }
}
