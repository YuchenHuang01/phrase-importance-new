#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-close paraphrase generator for KL experiments.
Strategy:
  1) Generate minimal, controlled edits (no API calls).
  2) Verify with strong guards:
     - ST cosine >= 0.86 (paraphrase-mpnet)
     - Bidirectional NLI = entailment (DeBERTa-MNLI)
     - Surface closeness: chrF >= 0.80, edit ratio >= 0.85
     - Keyword anchoring (protected tokens)
     - Optional POS/lemma checks (spaCy) + no-negation-flip
Outputs 3–5 very-close paraphrases or fewer if strict filters fail.
"""

from typing import List, Tuple, Dict, Optional, Iterable
import re
import itertools

# --- Optional deps (soft-fail) ---
try:
    import spacy
    _SPACY = True
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY = False
    _NLP = None

try:
    from rapidfuzz.distance import Levenshtein
    _LEV = True
except Exception:
    _LEV = False

try:
    import sacrebleu
    _SACRE = True
except Exception:
    _SACRE = False

# --- Hugging Face models (local; no API needed) ---
from sentence_transformers import SentenceTransformer, util as st_util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ------------------------------
# Minimal Edit Generator (rules)
# ------------------------------
class MinimalEditGenerator:
    """Produce tiny, meaning-preserving edits with constraints."""
    CONTRACTIONS = [
        ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
        ("you are", "you're"), ("I am", "I'm"), ("we are", "we're"),
        ("it is", "it's"), ("cannot", "can't"), ("should not", "shouldn't"),
    ]
    EXPANSIONS = [(b, a) for (a, b) in CONTRACTIONS]

    LEX_SUBS = {
        # conservative synonym set; keep short and precise
        "say so": ["state that", "make that clear", "say that"],
        "unsure": ["uncertain", "not sure"],
        "avoid": ["do not provide", "refrain from giving"],
        "advice": ["guidance"],
        "explicitly": ["clearly"],
        "about": ["regarding"],
        "when": ["if"],
        "you should": ["please", "you ought to"],
    }

    INSERTIONS = [
        # insert hedges/adverbs in safe positions
        ("you should", "you should explicitly"),
        ("state that", "clearly state that"),
        ("say so", "clearly say so"),
        ("do not provide", "do not explicitly provide"),
    ]

    def __init__(self, max_edits: int = 3):
        self.max_edits = max_edits

    def _apply_sub(self, s: str, fr: str, to: str) -> Optional[str]:
        # word-boundary conservative replacement
        pat = r'\b' + re.escape(fr) + r'\b'
        if re.search(pat, s, flags=re.IGNORECASE):
            return re.sub(pat, to, s, count=1, flags=re.IGNORECASE)
        return None

    def _variants_from_rules(self, s: str) -> Iterable[str]:
        # 1) contractions/expansions
        for a, b in itertools.chain(self.CONTRACTIONS, self.EXPANSIONS):
            t = self._apply_sub(s, a, b)
            if t and t != s:
                yield t

        # 2) lexical subs (one-at-a-time)
        lower = s.lower()
        for key, subs in self.LEX_SUBS.items():
            if re.search(r'\b' + re.escape(key) + r'\b', lower):
                for sub in subs:
                    t = self._apply_sub(s, key, sub)
                    if t and t != s:
                        yield t

        # 3) insertions (safe adverbs)
        for anchor, insertion in self.INSERTIONS:
            if re.search(r'\b' + re.escape(anchor) + r'\b', s, flags=re.IGNORECASE):
                t = self._apply_sub(s, anchor, insertion)
                if t and t != s:
                    yield t

        # 4) light rewrites: when/if, about/regarding, that insertion
        t = self._apply_sub(s, "when", "if")
        if t and t != s: yield t
        t = self._apply_sub(s, "about", "regarding")
        if t and t != s: yield t
        # add "that" after verbs like "say"/"state"
        t = re.sub(r'\bsay (so|it)\b', r'say that \1', s, count=1, flags=re.IGNORECASE)
        if t != s: yield t
        t = re.sub(r'\bstate (it)\b', r'state that \1', s, count=1, flags=re.IGNORECASE)
        if t != s: yield t

    def generate(self, s: str, k: int = 6) -> List[str]:
        seen = set([s.lower()])
        out = []
        for cand in self._variants_from_rules(s):
            lc = cand.lower()
            if lc not in seen:
                seen.add(lc)
                out.append(cand)
            if len(out) >= k:
                break
        return out


# ------------------------------
# Verifier (semantic + surface)
# ------------------------------
class ParaphraseVerifier:
    def __init__(
        self,
        st_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        cos_min: float = 0.86,
        nli_p: float = 0.90,
        chrf_min: float = 0.80,
        edit_min: float = 0.85,
        jaccard_lemmas_min: float = 0.70,
        enforce_pos_pattern: bool = False,
        protected_keywords: Optional[List[str]] = None,
    ):
        self.st = SentenceTransformer(st_model_name)
        self.nli_tok = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli.eval()
        self.cos_min = cos_min
        self.nli_p = nli_p
        self.chrf_min = chrf_min
        self.edit_min = edit_min
        self.jaccard_lemmas_min = jaccard_lemmas_min
        self.enforce_pos_pattern = enforce_pos_pattern
        self.protected_keywords = [k.lower() for k in (protected_keywords or [])]

    # --- Utilities ---
    def _cosine(self, a: str, b: str) -> float:
        ea = self.st.encode([a], normalize_embeddings=True)
        eb = self.st.encode([b], normalize_embeddings=True)
        return float(st_util.cos_sim(torch.tensor(ea), torch.tensor(eb))[0][0])

    @torch.no_grad()
    def _nli_entail_prob(self, prem: str, hyp: str) -> float:
        inputs = self.nli_tok(prem, hyp, return_tensors="pt", truncation=True)
        logits = self.nli(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        # MNLI label order: [contradiction, neutral, entailment]
        return float(probs[-1])

    def _bidirectional_entail(self, a: str, b: str) -> bool:
        return (self._nli_entail_prob(a, b) >= self.nli_p and
                self._nli_entail_prob(b, a) >= self.nli_p)

    def _edit_ratio(self, a: str, b: str) -> float:
        if not _LEV:
            return 1.0  # skip if missing, assume ok
        dist = Levenshtein.distance(a, b)
        maxlen = max(len(a), len(b))
        return 1.0 - (dist / maxlen if maxlen else 0.0)

    def _chrf(self, a: str, b: str) -> float:
        if not _SACRE:
            return 1.0
        return sacrebleu.CHRF(word_order=2).corpus_score([b], [[a]]).score / 100.0

    def _contains_all_protected(self, src: str, hyp: str) -> bool:
        if not self.protected_keywords:
            return True
        s = src.lower()
        h = hyp.lower()
        for kw in self.protected_keywords:
            if (kw in s) and (kw not in h):
                return False
        return True

    def _negation_flip(self, a: str, b: str) -> bool:
        neg = {"not", "never", "no", "cannot", "can't", "won't", "shouldn't", "doesn't", "don't", "didn't"}
        ta = set(w.lower() for w in re.findall(r"[A-Za-z']+", a))
        tb = set(w.lower() for w in re.findall(r"[A-Za-z']+", b))
        # true if one has negation and the other doesn't
        return (len(ta & neg) > 0) != (len(tb & neg) > 0)

    def _jaccard_lemmas(self, a: str, b: str) -> float:
        if not _SPACY:
            return 1.0
        da, db = _NLP(a), _NLP(b)
        la = {t.lemma_.lower() for t in da if t.is_alpha and not t.is_stop}
        lb = {t.lemma_.lower() for t in db if t.is_alpha and not t.is_stop}
        if not la and not lb:
            return 1.0
        return len(la & lb) / max(1, len(la | lb))

    def _pos_pattern_match(self, a: str, b: str) -> float:
        if not _SPACY:
            return 1.0
        da, db = _NLP(a), _NLP(b)
        pa = [t.pos_ for t in da if t.is_alpha]
        pb = [t.pos_ for t in db if t.is_alpha]
        # simple overlap
        common = sum(1 for x, y in itertools.zip_longest(pa, pb) if x == y)
        return common / max(1, max(len(pa), len(pb)))

    def verify(self, src: str, hyp: str) -> Tuple[bool, Dict[str, float]]:
        cos = self._cosine(src, hyp)
        if cos < self.cos_min:
            return False, {"cos": cos}

        if self._negation_flip(src, hyp):
            return False, {"cos": cos, "negation_flip": 1.0}

        if not self._contains_all_protected(src, hyp):
            return False, {"cos": cos, "protected_missing": 1.0}

        chrf = self._chrf(src, hyp)
        if chrf < self.chrf_min:
            return False, {"cos": cos, "chrf": chrf}

        edit_ratio = self._edit_ratio(src, hyp)
        if edit_ratio < self.edit_min:
            return False, {"cos": cos, "chrf": chrf, "edit_ratio": edit_ratio}

        jac = self._jaccard_lemmas(src, hyp)
        if jac < self.jaccard_lemmas_min:
            return False, {"cos": cos, "chrf": chrf, "edit_ratio": edit_ratio, "jaccard_lemmas": jac}

        if self.enforce_pos_pattern:
            pos_match = self._pos_pattern_match(src, hyp)
            if pos_match < 0.90:
                return False, {"cos": cos, "chrf": chrf, "edit_ratio": edit_ratio, "jaccard_lemmas": jac, "pos_match": pos_match}

        # Strong semantic equivalence via NLI (both directions)
        if not self._bidirectional_entail(src, hyp):
            return False, {"cos": cos, "chrf": chrf, "edit_ratio": edit_ratio, "jaccard_lemmas": jac, "nli": 0.0}

        return True, {"cos": cos, "chrf": chrf, "edit_ratio": edit_ratio, "jaccard_lemmas": jac}


# ------------------------------
# Orchestrator
# ------------------------------
class UltraCloseParaphraser:
    def __init__(
        self,
        protected_keywords: Optional[List[str]] = None,
        cos_min: float = 0.86,
        nli_p: float = 0.90,
        chrf_min: float = 0.80,
        edit_min: float = 0.85,
        jaccard_lemmas_min: float = 0.70,
        enforce_pos_pattern: bool = False,
        max_edits: int = 3,
    ):
        self.generator = MinimalEditGenerator(max_edits=max_edits)
        self.verifier = ParaphraseVerifier(
            cos_min=cos_min,
            nli_p=nli_p,
            chrf_min=chrf_min,
            edit_min=edit_min,
            jaccard_lemmas_min=jaccard_lemmas_min,
            enforce_pos_pattern=enforce_pos_pattern,
            protected_keywords=protected_keywords or [],
        )

    def paraphrase(self, s: str, k: int = 5) -> List[Tuple[str, Dict[str, float]]]:
        cands = self.generator.generate(s, k=12)  # generate a few more than needed
        out = []
        seen = set()
        for cand in cands:
            lc = cand.lower().strip()
            if lc in seen or lc == s.lower().strip():
                continue
            ok, metrics = self.verifier.verify(s, cand)
            if ok:
                out.append((cand, metrics))
            if len(out) >= k:
                break
            seen.add(lc)
        return out


# ------------------------------
# File I/O functions
# ------------------------------
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
    
    print(f"Loaded {len(phrases)} phrases for high-similarity paraphrase generation")
    return phrases

def save_high_similarity_paraphrases(results: Dict[str, List[Tuple[str, Dict[str, float]]]], output_file: str):
    """Save high-similarity paraphrases with strict filtering."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("HIGH-SIMILARITY PARAPHRASES FOR KL DIVERGENCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Generated using ultra-close paraphrase generation with strict filtering:\n")
        f.write("- ST cosine similarity >= 0.86 (paraphrase-mpnet-base-v2)\n")
        f.write("- Bidirectional NLI entailment >= 0.90 (DeBERTa-MNLI)\n")
        f.write("- chrF score >= 0.80\n")
        f.write("- Edit distance ratio >= 0.85\n")
        f.write("- Jaccard lemma overlap >= 0.70\n")
        f.write("- Protected keyword preservation\n")
        f.write("- No negation flips\n")
        f.write("Format: Original phrase B -> High-similarity variants {B_1, B_2, ..., B_k}\n")
        f.write("-" * 60 + "\n\n")
        
        total_variants = 0
        successful_phrases = 0
        
        for i, (original, pairs) in enumerate(results.items(), 1):
            f.write(f"{i:3d}. ORIGINAL: \"{original}\"\n")
            
            if pairs:
                f.write(f"     HIGH-SIMILARITY VARIANTS:\n")
                for j, (variant, metrics) in enumerate(pairs, 1):
                    cos = metrics.get('cos', 0.0)
                    chrf = metrics.get('chrf', 0.0)
                    edit_ratio = metrics.get('edit_ratio', 0.0)
                    f.write(f"       {j}. \"{variant}\" (cos: {cos:.3f}, chrF: {chrf:.3f}, edit: {edit_ratio:.3f})\n")
                total_variants += len(pairs)
                successful_phrases += 1
            else:
                f.write(f"     No variants passed strict filters (consider relaxing cos_min to 0.84)\n")
            
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total original phrases: {len(results)}\n")
        f.write(f"  Phrases with variants: {successful_phrases}\n")
        f.write(f"  Total variants generated: {total_variants}\n")
        if successful_phrases > 0:
            f.write(f"  Average variants per successful phrase: {total_variants/successful_phrases:.1f}\n")
        f.write(f"\nFor KL analysis: Replace each B with ultra-close variants B_i\n")
        f.write(f"to measure fine-grained semantic sensitivity.\n")

# ------------------------------
# Main execution with file I/O
# ------------------------------
if __name__ == "__main__":
    import sys
    
    # Get input file from command line or use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "candidate_phrases_no_overlap.txt"
    output_file = "high_similarity_paraphrases.txt"
    
    print("Ultra-Close Paraphrase Generator for KL Analysis")
    print("=" * 50)
    print("Using strict filtering for minimal meaning drift:")
    print("- ST cosine >= 0.86, NLI entailment >= 0.90")
    print("- chrF >= 0.80, edit ratio >= 0.85")
    print("- Protected keywords preserved, no negation flips")
    print()
    
    try:
        # Load phrases
        phrases = load_phrases_from_file(input_file)
        
        # Initialize paraphraser with strict settings
        paraphraser = UltraCloseParaphraser(
            protected_keywords=["legal", "medical", "financial", "AI", "assistant", "Claude", "Anthropic"],
            cos_min=0.86,
            nli_p=0.90,
            chrf_min=0.80,
            edit_min=0.85,
            jaccard_lemmas_min=0.70,
            enforce_pos_pattern=False,
            max_edits=3,
        )
        
        # Generate paraphrases for all phrases
        results = {}
        print(f"Generating high-similarity paraphrases for {len(phrases)} phrases...")
        
        for i, phrase in enumerate(phrases):
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(phrases)} phrases...")
            
            pairs = paraphraser.paraphrase(phrase, k=5)
            results[phrase] = pairs
        
        # Save results
        save_high_similarity_paraphrases(results, output_file)
        
        # Print summary
        total_phrases = len(results)
        phrases_with_variants = sum(1 for pairs in results.values() if pairs)
        total_variants = sum(len(pairs) for pairs in results.values())
        
        print(f"\nHIGH-SIMILARITY PARAPHRASE GENERATION COMPLETE:")
        print(f"  Phrases processed: {total_phrases}")
        print(f"  Phrases with variants: {phrases_with_variants} ({phrases_with_variants/total_phrases*100:.1f}%)")
        print(f"  Total variants generated: {total_variants}")
        print(f"  Results saved to: {output_file}")
        
        # Show a few examples
        print(f"\nExample high-similarity paraphrases:")
        example_phrases = list(results.items())[:3]
        for original, pairs in example_phrases:
            print(f"\nOriginal: \"{original}\"")
            if pairs:
                for i, (variant, metrics) in enumerate(pairs[:3], 1):
                    cos = metrics.get('cos', 0.0)
                    chrf = metrics.get('chrf', 0.0)
                    edit_ratio = metrics.get('edit_ratio', 0.0)
                    print(f"  {i}. \"{variant}\" (cos: {cos:.3f}, chrF: {chrf:.3f}, edit: {edit_ratio:.3f})")
            else:
                print(f"  No variants passed strict filters")
        
        print(f"\nReady for ultra-sensitive KL divergence analysis!")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        print("Usage: python generate_paraphrases_higher_similarity.py [input_file]")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
