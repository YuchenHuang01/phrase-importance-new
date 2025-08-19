#!/usr/bin/env python3
"""
Adaptive paraphrase generator with practical filtering approach.
Uses multi-stage filtering with adaptive strictness to avoid all-or-nothing outcomes.
"""

import re
import torch
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import itertools

# Optional dependencies
try:
    import spacy
    _SPACY = True
    _NLP = spacy.load("en_core_web_sm")
except:
    _SPACY = False
    _NLP = None

try:
    from rapidfuzz.distance import Levenshtein
    _LEV = True
except:
    _LEV = False


class AdaptiveParaphraseGenerator:
    def __init__(
        self,
        paraphrase_model: str = "eugenesiow/bart-paraphrase",
        rerank_model: str = "sentence-transformers/all-mpnet-base-v2", 
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        use_local_paraphraser: bool = True,
        protected_keywords: List[str] = None
    ):
        # Models
        self.rerank_encoder = SentenceTransformer(rerank_model)
        
        # NLI model for verification
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.eval()
        
        # Optional paraphrase model
        if use_local_paraphraser:
            try:
                self.paraphraser = pipeline(
                    "text2text-generation",
                    model=paraphrase_model,
                    tokenizer=paraphrase_model,
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                print(f"Warning: Could not load {paraphrase_model}, using rule-based generation only")
                self.paraphraser = None
        else:
            self.paraphraser = None
            
        self.protected_keywords = [kw.lower() for kw in (protected_keywords or [])]
        
        # Rule-based edits for minimal changes
        self.minimal_edits = [
            ("do not", "don't"),
            ("does not", "doesn't"), 
            ("you are", "you're"),
            ("it is", "it's"),
            ("cannot", "can't"),
            ("should not", "shouldn't"),
            ("will not", "won't"),
            ("in order to", "to"),
            ("you should", "please"),
            ("you must", "you need to"),
            ("when", "if"),
            ("about", "regarding"),
            ("say so", "state that"),
            ("unsure", "uncertain"),
            ("advice", "guidance"),
            ("explicitly", "clearly"),
        ]
    
    def generate_candidates(self, text: str, num_candidates: int = 16) -> List[str]:
        """Generate paraphrase candidates using multiple methods."""
        candidates = set()
        
        # Method 1: Rule-based minimal edits
        rule_candidates = self._generate_rule_based(text)
        candidates.update(rule_candidates)
        
        # Method 2: Model-based paraphrasing (if available)
        if self.paraphraser:
            model_candidates = self._generate_model_based(text, num_candidates // 2)
            candidates.update(model_candidates)
        
        # Remove duplicates and original
        candidates.discard(text)
        candidates.discard(text.lower())
        
        return list(candidates)[:num_candidates]
    
    def _generate_rule_based(self, text: str) -> List[str]:
        """Generate candidates using minimal rule-based edits."""
        candidates = []
        
        # Single edits
        for old, new in self.minimal_edits:
            if re.search(r'\b' + re.escape(old) + r'\b', text, re.IGNORECASE):
                candidate = re.sub(r'\b' + re.escape(old) + r'\b', new, text, count=1, flags=re.IGNORECASE)
                if candidate != text:
                    candidates.append(candidate)
        
        # Reverse edits
        for new, old in self.minimal_edits:
            if re.search(r'\b' + re.escape(old) + r'\b', text, re.IGNORECASE):
                candidate = re.sub(r'\b' + re.escape(old) + r'\b', new, text, count=1, flags=re.IGNORECASE)
                if candidate != text:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_model_based(self, text: str, num_candidates: int = 8) -> List[str]:
        """Generate candidates using a paraphrase model."""
        if not self.paraphraser:
            return []
        
        try:
            # Generate with low temperature for minimal changes
            outputs = self.paraphraser(
                text,
                max_length=len(text.split()) + 10,
                min_length=max(1, len(text.split()) - 3),
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=num_candidates,
                do_sample=True
            )
            
            candidates = []
            for output in outputs:
                candidate = output['generated_text'].strip()
                if candidate != text:
                    candidates.append(candidate)
            
            return candidates
        except Exception as e:
            print(f"Warning: Model-based generation failed: {e}")
            return []
    
    def rerank_by_similarity(self, text: str, candidates: List[str], 
                           cos_min: float = 0.90, cos_max: float = 0.98) -> List[Tuple[str, float]]:
        """Rerank candidates by cosine similarity within tight bounds."""
        if not candidates:
            return []
        
        # Get embeddings
        all_texts = [text] + candidates
        embeddings = self.rerank_encoder.encode(all_texts, normalize_embeddings=True)
        
        original_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        # Calculate similarities
        similarities = torch.cosine_similarity(
            torch.tensor(original_emb).unsqueeze(0),
            torch.tensor(candidate_embs),
            dim=1
        ).numpy()
        
        # Filter by similarity bounds
        filtered = []
        for candidate, sim in zip(candidates, similarities):
            if cos_min <= sim <= cos_max:
                # Check length constraint (±10%)
                orig_len = len(text.split())
                cand_len = len(candidate.split())
                if abs(cand_len - orig_len) / orig_len <= 0.10:
                    # Check protected keywords preserved
                    if self._check_protected_keywords(text, candidate):
                        filtered.append((candidate, float(sim)))
        
        # Sort by similarity (descending)
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered
    
    def _check_protected_keywords(self, original: str, candidate: str) -> bool:
        """Check that protected keywords are preserved."""
        if not self.protected_keywords:
            return True
        
        orig_lower = original.lower()
        cand_lower = candidate.lower()
        
        for keyword in self.protected_keywords:
            if keyword in orig_lower and keyword not in cand_lower:
                return False
        return True
    
    @torch.no_grad()
    def verify_with_nli(self, original: str, candidate: str, 
                       max_contradiction: float = 0.10,
                       min_entailment: float = 0.80,
                       fallback_cosine: float = 0.93) -> Tuple[bool, Dict[str, float]]:
        """Two-gate NLI verification with fallback."""
        
        # Get NLI predictions
        inputs = self.nli_tokenizer(original, candidate, return_tensors="pt", truncation=True)
        outputs = self.nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        
        # Assuming standard NLI labels: [contradiction, neutral, entailment]
        contradiction_prob = float(probs[0])
        entailment_prob = float(probs[2])
        
        metrics = {
            "contradiction": contradiction_prob,
            "entailment": entailment_prob
        }
        
        # Gate A: Safety check (low contradiction)
        if contradiction_prob > max_contradiction:
            return False, metrics
        
        # Gate B: Semantic closeness (entailment OR high cosine)
        if entailment_prob >= min_entailment:
            return True, metrics
        
        # Fallback: check cosine similarity
        embs = self.rerank_encoder.encode([original, candidate], normalize_embeddings=True)
        cosine = torch.cosine_similarity(
            torch.tensor(embs[0]).unsqueeze(0),
            torch.tensor(embs[1]).unsqueeze(0)
        ).item()
        
        metrics["fallback_cosine"] = cosine
        
        if cosine >= fallback_cosine:
            return True, metrics
        
        return False, metrics
    
    def generate_paraphrases(self, text: str, target_count: int = 3) -> List[Tuple[str, Dict[str, float]]]:
        """Generate paraphrases with adaptive strictness."""
        
        # Stage 1: Generate candidates
        candidates = self.generate_candidates(text, num_candidates=24)
        if not candidates:
            return []
        
        # Stage 2: Multi-level adaptive filtering
        strictness_levels = [
            {"cos_min": 0.95, "cos_max": 0.99, "max_contra": 0.05, "min_entail": 0.85},
            {"cos_min": 0.93, "cos_max": 0.98, "max_contra": 0.08, "min_entail": 0.80},
            {"cos_min": 0.90, "cos_max": 0.97, "max_contra": 0.10, "min_entail": 0.75},
        ]
        
        results = []
        
        for level in strictness_levels:
            # Rerank by cosine similarity
            reranked = self.rerank_by_similarity(
                text, candidates, 
                cos_min=level["cos_min"], 
                cos_max=level["cos_max"]
            )
            
            # Verify with NLI
            for candidate, cosine in reranked:
                if len(results) >= target_count:
                    break
                
                is_valid, metrics = self.verify_with_nli(
                    text, candidate,
                    max_contradiction=level["max_contra"],
                    min_entailment=level["min_entail"],
                    fallback_cosine=0.93
                )
                
                if is_valid:
                    metrics["cosine"] = cosine
                    results.append((candidate, metrics))
            
            # If we have enough results, stop
            if len(results) >= target_count:
                break
        
        return results[:target_count]


def load_phrases_from_file(filepath: str) -> List[str]:
    """Load phrases from candidate file."""
    phrases = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('"') and '" |' in line:
            phrase = line.split('" |')[0][1:]
            phrases.append(phrase)
    
    print(f"Loaded {len(phrases)} phrases for adaptive paraphrase generation")
    return phrases


def save_adaptive_paraphrases(results: Dict[str, List[Tuple[str, Dict[str, float]]]], output_file: str):
    """Save adaptively generated paraphrases."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ADAPTIVE HIGH-SIMILARITY PARAPHRASES FOR KL DIVERGENCE ANALYSIS\n")
        f.write("=" * 65 + "\n\n")
        f.write("Generated using adaptive multi-stage filtering:\n")
        f.write("- Stage 1: Rule-based + model-based candidate generation\n")
        f.write("- Stage 2: Cosine similarity filtering (0.90-0.98 range)\n")
        f.write("- Stage 3: Two-gate NLI verification with adaptive strictness\n")
        f.write("- Protected keyword preservation + length constraints\n")
        f.write("Format: Original phrase -> Adaptive paraphrases\n")
        f.write("-" * 65 + "\n\n")
        
        total_variants = 0
        successful_phrases = 0
        
        for i, (original, pairs) in enumerate(results.items(), 1):
            f.write(f"{i:3d}. ORIGINAL: \"{original}\"\n")
            
            if pairs:
                f.write(f"     ADAPTIVE PARAPHRASES:\n")
                for j, (variant, metrics) in enumerate(pairs, 1):
                    cosine = metrics.get('cosine', 0.0)
                    entailment = metrics.get('entailment', 0.0)
                    contradiction = metrics.get('contradiction', 0.0)
                    f.write(f"       {j}. \"{variant}\"\n")
                    f.write(f"           cos: {cosine:.3f}, entail: {entailment:.3f}, contra: {contradiction:.3f}\n")
                total_variants += len(pairs)
                successful_phrases += 1
            else:
                f.write(f"     No paraphrases generated (consider relaxing further)\n")
            
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total original phrases: {len(results)}\n")
        f.write(f"  Phrases with variants: {successful_phrases}\n")
        f.write(f"  Total variants generated: {total_variants}\n")
        if successful_phrases > 0:
            f.write(f"  Average variants per successful phrase: {total_variants/successful_phrases:.1f}\n")


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "candidate_phrases_no_overlap.txt"
    output_file = "adaptive_high_similarity_paraphrases.txt"
    
    print("Adaptive High-Similarity Paraphrase Generator")
    print("=" * 45)
    print("Using multi-stage adaptive filtering:")
    print("- Rule-based + model-based generation")
    print("- Cosine similarity reranking (0.90-0.98)")
    print("- Two-gate NLI verification with fallbacks")
    print("- Adaptive strictness levels")
    print()
    
    try:
        # Load phrases
        phrases = load_phrases_from_file(input_file)
        
        # Initialize generator
        generator = AdaptiveParaphraseGenerator(
            protected_keywords=["legal", "medical", "financial", "AI", "assistant", "Claude", "Anthropic"],
            use_local_paraphraser=True  # Enable BART paraphrase model
        )
        
        # Generate paraphrases
        results = {}
        print(f"Generating adaptive paraphrases for {len(phrases)} phrases...")
        
        for i, phrase in enumerate(phrases):
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(phrases)} phrases...")
            
            paraphrases = generator.generate_paraphrases(phrase, target_count=3)
            results[phrase] = paraphrases
        
        # Save results
        save_adaptive_paraphrases(results, output_file)
        
        # Print summary
        total_phrases = len(results)
        phrases_with_variants = sum(1 for pairs in results.values() if pairs)
        total_variants = sum(len(pairs) for pairs in results.values())
        
        print(f"\nADAPTIVE PARAPHRASE GENERATION COMPLETE:")
        print(f"  Phrases processed: {total_phrases}")
        print(f"  Phrases with variants: {phrases_with_variants} ({phrases_with_variants/total_phrases*100:.1f}%)")
        print(f"  Total variants generated: {total_variants}")
        print(f"  Results saved to: {output_file}")
        
        # Show examples
        print(f"\nExample adaptive paraphrases:")
        example_phrases = list(results.items())[:3]
        for original, pairs in example_phrases:
            print(f"\nOriginal: \"{original}\"")
            if pairs:
                for i, (variant, metrics) in enumerate(pairs, 1):
                    cosine = metrics.get('cosine', 0.0)
                    entailment = metrics.get('entailment', 0.0)
                    print(f"  {i}. \"{variant}\" (cos: {cosine:.3f}, entail: {entailment:.3f})")
            else:
                print(f"  No variants generated")
                
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()