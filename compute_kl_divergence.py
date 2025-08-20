#!/usr/bin/env python3
"""
Compute sequence-level KL divergence for each system prompt independently,
using shared samples from the original prompt (P) and teacher-forced
scoring under paraphrased prompts (Q_i).

Edits:
- Fix Qwen generate() crash: no past_key_values=None; use_cache=True; trust_remote_code=True.
- Explicit pad/eos token handling; send tensors to model.device.
- Robust phrase normalization for paraphrase lookups.
- Keep original shared-sample KL estimator & caching.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import pickle
import re
import hashlib
import random
import unicodedata

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compute_kl_divergence.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ------------------------
# Utils
# ------------------------
def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _norm(s: str) -> str:
    """Normalize phrases so keys match despite casing/spacing/punctuation."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .,!?:;\"'")
    return s

# ------------------------
# Main class
# ------------------------
class KLDivergenceComputer:
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-base",
                 device: str = None,
                 torch_dtype: Any = None,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 num_samples: int = 8,
                 max_new_tokens: int = 150,
                 replace_count: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.num_samples = int(num_samples)
        self.max_new_tokens = int(max_new_tokens)
        self.replace_count = int(replace_count)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")

        # Choose dtype
        if torch_dtype is None and torch.cuda.is_available():
            torch_dtype = torch.float16

        # ---- Tokenizer & Model (Qwen-safe) ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,   # needed for Qwen; harmless otherwise
            use_fast=False            # Qwen often uses slow tokenizer
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(torch_dtype if torch_dtype else None),
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Ensure pad/eos consistency
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model, "generation_config", None) and self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # Data storage
        self.all_candidate_phrases: List[str] = []
        self.paraphrases: Dict[str, List[str]] = {}
        self.system_prompts: List[Dict[str, str]] = []
        self.user_prompts: List[str] = []
        self.results_by_prompt: Dict[str, Any] = {}

        # Cache: key -> (seqs, logp_P)
        self._sample_cache: Dict[str, Tuple[List[torch.Tensor], np.ndarray]] = {}

    # ------------------------
    # Loading
    # ------------------------
    def load_data(self, phrases_file: str, paraphrases_file: str, prompts_file: str, user_prompts_file: str):
        logger.info("Loading data files...")

        # Candidate phrases: "..." | N words | prompt_k
        with open(phrases_file, 'r', encoding='utf-8') as f:
            content = f.read()
            pattern = r'"([^"]+)"\s*\|\s*\d+\s*words?\s*\|\s*prompt_\d+'
            self.all_candidate_phrases = re.findall(pattern, content)
            logger.info(f"Loaded {len(self.all_candidate_phrases)} candidate phrases")

        # Paraphrases: ORIGINAL + numbered paraphrases
        with open(paraphrases_file, 'r', encoding='utf-8') as f:
            content = f.read()
            current_original = None
            temp_map: Dict[str, List[str]] = {}
            for line in content.splitlines():
                if 'ORIGINAL:' in line:
                    m = re.search(r'ORIGINAL:\s*"([^"]+)"', line)
                    if m:
                        current_original = m.group(1)
                        temp_map[current_original] = []
                elif current_original and '"' in line and re.search(r'\s+\d+\.\s*"([^"]+)"', line):
                    m = re.search(r'\s+\d+\.\s*"([^"]+)"', line)
                    if m:
                        temp_map[current_original].append(m.group(1))
        # Normalize keys for robust matching
        self.paraphrases = { _norm(k): v for k, v in temp_map.items() }
        logger.info(f"Loaded paraphrases for {len(self.paraphrases)} phrases")

        # System prompts (first 34)
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split('================================================================================')
            self.system_prompts = []
            for ch in chunks:
                ch = ch.strip()
                if not ch or not ch.startswith('Prompt'):
                    continue
                lines = ch.splitlines()
                if len(lines) > 1:
                    m = re.search(r'Prompt\s+(\d+)', lines[0])
                    if not m:
                        continue
                    num = int(m.group(1))
                    if num > 34:
                        continue
                    prompt_text = '\n'.join(lines[1:]).strip()
                    self.system_prompts.append({'id': f'prompt_{num}', 'text': prompt_text})
        logger.info(f"Loaded {len(self.system_prompts)} system prompts (first 34).")

        # User prompts JSON
        with open(user_prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "all_prompts" in data and isinstance(data["all_prompts"], list):
                    self.user_prompts = list(data["all_prompts"])
                elif "prompts_by_category" in data and isinstance(data["prompts_by_category"], dict):
                    flat = []
                    for v in data["prompts_by_category"].values():
                        flat.extend(v)
                    self.user_prompts = flat
                else:
                    flat = []
                    for v in data.values():
                        if isinstance(v, list):
                            flat.extend(v)
                    self.user_prompts = flat
            elif isinstance(data, list):
                self.user_prompts = data
            else:
                raise ValueError("user_prompts.json format not recognized.")
        logger.info(f"Loaded {len(self.user_prompts)} user prompts.")

    # ------------------------
    # Helpers
    # ------------------------
    def find_phrases_in_prompt(self, prompt_text: str) -> List[str]:
        """Return candidate phrases that appear in this prompt (raw strings)."""
        return [p for p in self.all_candidate_phrases if p in prompt_text]

    def replace_phrase_in_prompt(self, prompt: str, original_phrase: str, replacement_phrase: str, count: int = None) -> str:
        c = self.replace_count if count is None else count
        return re.sub(re.escape(original_phrase), replacement_phrase, prompt, count=c)

    def _prompt_prefix(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        h = hashlib.sha1()
        h.update(system_prompt.encode('utf-8')); h.update(b'\x00'); h.update(user_prompt.encode('utf-8'))
        return f"{h.hexdigest()}|M={self.num_samples}|N={self.max_new_tokens}|T={self.temperature}|p={self.top_p}"

    # ------------------------
    # Sampling & Scoring
    # ------------------------
    @torch.inference_mode()
    def sample_sequences_from_P(self, system_prompt: str, user_prompt: str) -> Tuple[List[torch.Tensor], np.ndarray]:
        """Sample M continuations y~P(.|S,x), return (list of cont token tensors, logP list)."""
        cache_key = self._cache_key(system_prompt, user_prompt)
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        prefix = self._prompt_prefix(system_prompt, user_prompt)
        inpt = self.tokenizer(prefix, return_tensors="pt")
        inpt = {k: v.to(self.model.device) for k, v in inpt.items()}

        seqs: List[torch.Tensor] = []
        logp_list: List[float] = []

        for _ in range(self.num_samples):
            gen = self.model.generate(
                **inpt,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,  # important for Qwen
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # Continuation only (generated part)
            cont_ids = gen.sequences[0][inpt["input_ids"].shape[1]:]  # [L]
            # scores: list of [1, V] across steps
            step_logprobs = torch.log_softmax(torch.stack(gen.scores, dim=0), dim=-1)  # [L, 1, V]
            token_logps = step_logprobs.squeeze(1).gather(-1, cont_ids.view(-1, 1)).squeeze(-1)  # [L]
            seqs.append(cont_ids.detach().cpu())
            logp_list.append(float(token_logps.sum().item()))

        logp_arr = np.array(logp_list, dtype=np.float64)
        self._sample_cache[cache_key] = (seqs, logp_arr)
        return seqs, logp_arr

    @torch.inference_mode()
    def score_sequences_under_Q(self, system_prompt_Q: str, user_prompt: str, continuations: List[torch.Tensor]) -> np.ndarray:
        """Teacher-forced log Q(y|S_Q,x) over the continuation region only."""
        prefix = self._prompt_prefix(system_prompt_Q, user_prompt)
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")
        prefix_ids = {k: v.to(self.model.device) for k, v in prefix_ids.items()}
        prefix_ids = prefix_ids["input_ids"][0]

        logps = []
        for cont_cpu in continuations:
            cont = cont_cpu.to(self.model.device)
            ids = torch.cat([prefix_ids, cont], dim=0).unsqueeze(0)  # [1, P+L]
            outputs = self.model(ids)
            logits = outputs.logits[:, :-1, :]
            tgt = ids[:, 1:]

            # continuation region: predictions that correspond to continuation tokens
            cont_start = prefix_ids.shape[0] - 1
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            mask[:, cont_start:] = True

            logprobs_all = torch.log_softmax(logits, dim=-1)
            token_logps = logprobs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, P+L-1]
            seq_logp = token_logps[mask].sum().item()
            logps.append(seq_logp)

        return np.array(logps, dtype=np.float64)

    def kl_from_shared_samples(self, logp: np.ndarray, logq: np.ndarray) -> float:
        """KL(P||Q) ≈ mean[ logP - logQ ] with shared samples."""
        assert logp.shape == logq.shape
        return float(np.mean(logp - logq))

    # ------------------------
    # Main computation
    # ------------------------
    def compute_phrase_importance_per_prompt(self, save_every_prompts: int = 1, output_stub: str = "intermediate_results"):
        logger.info("Starting sequence-KL computation (shared samples) for each system prompt...")

        for prompt_idx, prompt_info in enumerate(self.system_prompts):
            prompt_id = prompt_info['id']
            prompt_text = prompt_info['text']
            logger.info(f"\nProcessing {prompt_id} ...")

            phrases_in_prompt = self.find_phrases_in_prompt(prompt_text)
            if not phrases_in_prompt:
                logger.warning(f"No candidate phrases found in {prompt_id}")
                continue
            logger.info(f"Found {len(phrases_in_prompt)} phrases in {prompt_id}")

            prompt_results = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'phrase_results': {}
            }

            pbar = tqdm(total=len(phrases_in_prompt), desc=f"KL for {prompt_id}")
            for phrase in phrases_in_prompt:
                key = _norm(phrase)
                plist = self.paraphrases.get(key, [])
                if not plist:
                    logger.warning(f"No paraphrases found for: {phrase}")
                    pbar.update(1)
                    continue

                user_prompt_scores: List[float] = []

                for user_prompt in self.user_prompts:
                    seqs, logp_P = self.sample_sequences_from_P(prompt_text, user_prompt)

                    paraphrase_kls: List[float] = []
                    for repl in plist:
                        S_mod = self.replace_phrase_in_prompt(prompt_text, phrase, repl, count=self.replace_count)
                        logq = self.score_sequences_under_Q(S_mod, user_prompt, seqs)
                        kl = self.kl_from_shared_samples(logp_P, logq)
                        paraphrase_kls.append(kl)

                    avg_paraphrase_kl = float(np.mean(paraphrase_kls)) if paraphrase_kls else 0.0
                    user_prompt_scores.append(avg_paraphrase_kl)

                avg_kl_score = float(np.mean(user_prompt_scores)) if user_prompt_scores else 0.0

                prompt_results['phrase_results'][phrase] = {
                    'avg_kl_score': avg_kl_score,
                    'num_paraphrases': len(plist),
                    'user_prompt_scores': [float(s) for s in user_prompt_scores],
                    'num_user_prompts': len(self.user_prompts)
                }

                pbar.update(1)

            pbar.close()

            # Sort phrases by KL score for this prompt
            sorted_items = sorted(
                prompt_results['phrase_results'].items(),
                key=lambda x: x[1]['avg_kl_score'],
                reverse=True
            )
            prompt_results['ranked_phrases'] = [
                {'rank': i + 1, 'phrase': phr, 'kl_score': float(dat['avg_kl_score'])}
                for i, (phr, dat) in enumerate(sorted_items)
            ]

            self.results_by_prompt[prompt_id] = prompt_results

            # Save periodically
            if (prompt_idx + 1) % save_every_prompts == 0:
                self.save_results(f"results/{output_stub}_{prompt_idx+1}.pkl")

        logger.info("\nAll prompts completed.")

    # ------------------------
    # Saving
    # ------------------------
    def save_results(self, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Saving results to {output_file}")

        with open(output_file, 'wb') as f:
            pickle.dump(self.results_by_prompt, f)

        summary = {}
        for pid, data in self.results_by_prompt.items():
            all_phrases = [
                {'rank': item['rank'], 'phrase': item['phrase'], 'kl_score': float(item['kl_score'])}
                for item in data.get('ranked_phrases', [])
            ]
            summary[pid] = {
                'num_phrases': len(data.get('phrase_results', {})),
                'ranked_phrases': all_phrases
            }

        json_file = re.sub(r'\.pkl$', '_summary.json', output_file)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote: {output_file} and {json_file}")

# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Compute sequence-KL for phrase importance per system prompt (shared samples)")
    ap.add_argument('--phrases', default='setup_data/candidate_phrases_no_overlap.txt', help='Path to candidate phrases file')
    ap.add_argument('--paraphrases', default='setup_data/simple_paraphrases.txt', help='Path to paraphrases file')
    ap.add_argument('--prompts', default='setup_data/system_prompts_generated.txt', help='Path to system prompts file')
    ap.add_argument('--user-prompts', default='prompts/user_prompts.json', help='Path to user prompts JSON file')
    ap.add_argument('--model', default='deepseek-ai/deepseek-llm-7b-base', help='HF model name (Qwen ok too)')
    ap.add_argument('--output', default='results/kl_divergence_results_per_prompt.pkl', help='Output pickle path')
    ap.add_argument('--num-samples', type=int, default=50, help='Samples per (P, user_prompt)')
    ap.add_argument('--max-new-tokens', type=int, default=150, help='Max new tokens to generate per sample')
    ap.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for generation under P')
    ap.add_argument('--top-p', type=float, default=0.9, help='Top-p nucleus for generation under P')
    ap.add_argument('--replace-count', type=int, default=1, help='Occurrences of phrase to replace in system prompt')
    ap.add_argument('--seed', type=int, default=42, help='Global RNG seed')
    return ap.parse_args()

def main():
    args = parse_args()
    set_global_seeds(args.seed)

    os.makedirs('results', exist_ok=True)

    computer = KLDivergenceComputer(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        replace_count=args.replace_count,
    )

    computer.load_data(args.phrases, args.paraphrases, args.prompts, args.user_prompts)
    computer.compute_phrase_importance_per_prompt(save_every_prompts=1, output_stub='intermediate_results')
    computer.save_results(args.output)
    logger.info("Computation complete!")

if __name__ == "__main__":
    main()
