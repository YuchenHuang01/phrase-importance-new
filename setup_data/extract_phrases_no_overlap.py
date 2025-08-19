#!/usr/bin/env python3
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

# ----------------------------------------
# Phrase extraction with overlap suppression
# ----------------------------------------

class PhraseExtractor:
    def __init__(self, jaccard_thresh: float = 0.85):
        self.jaccard_thresh = jaccard_thresh

        # --- your original pattern sets (unchanged) ---
        self.identity_patterns = [
            r'You are ([^.]+?(?:created by|developed by|trained by|designed by|built by)[^.]+)',
            r'You are ([^,]+, (?:an?|the) (?:AI|assistant|language model)[^.]*)',
            r'((?:AI|intelligent) assistant (?:created|developed|designed|built|trained) by [^.,]+)',
            r'(large language model (?:trained|developed) by [^.,]+)',
            r'You are ([^.]+AI[^.]*(?:created|developed|designed)[^.]*)',
        ]
        self.personality_patterns = [
            r'You are ([^.]*helpful[^.]*and[^.]*)',
            r'are ([^.]*helpful[^.]*,.*?(?:and|or)[^.]*)',
            r'(helpful,?\s+(?:honest|accurate|reliable|professional|thorough)[^.]*)',
            r'(thorough,?\s+analytical,?\s+and\s+reliable)',
            r'(curious,?\s+insightful,?\s+and\s+engaging)',
            r'(supportive,?\s+patient,?\s+and\s+methodical)',
            r'(knowledgeable,?\s+balanced,?\s+and\s+ethical)',
            r'(courteous,?\s+diplomatic,?\s+and\s+considerate)',
            r'(analytical,?\s+creative,?\s+and\s+solution-oriented)',
        ]
        self.behavior_patterns = [
            r'You should ([^.]{15,})',
            r'should ([^.]{15,})',
            r'You must ([^.]{15,})',
            r'must ([^.]{15,})',
            r'(provide [^.]{10,})',
            r'(encourage [^.]{10,})',
            r'(structure [^.]{10,})',
            r'(adapt [^.]{10,})',
            r'(help [^.]{15,})',
            r'(present [^.]{10,})',
            r'(promote [^.]{10,})',
            r'(synthesize [^.]{10,})',
            r'(approach [^.]{15,})',
        ]
        self.constraint_patterns = [
            r'(avoid [^.]{8,})',
            r'(Do not [^.]{8,})',
            r'(should not [^.]{8,})',
            r'(must not [^.]{8,})',
            r'(without [^.]{8,})',
            r'(refuse [^.]{8,})',
            r'(acknowledge [^.]{15,}limitations[^.]*)',
            r'(clearly state [^.]{8,})',
            r'(distinguish between [^.]{8,})',
            r'If you ([^.]{10,})',
            r'When ([^.]{10,})',
            r'unless ([^.]{8,})',
        ]
        self.technical_patterns = [
            r'(using [^.]{8,})',
            r'(with [^.]{15,}methodology[^.]*)',
            r'(through [^.]{10,})',
            r'(by [^.]{15,})',
            r'(based on [^.]{8,})',
            r'(according to [^.]{8,})',
            r'(following [^.]{8,})',
            r'(employing [^.]{8,})',
            r'(utilizing [^.]{8,})',
            r'(maintaining [^.]{8,})',
        ]
        self.domain_patterns = [
            r'(legal,?\s+medical,?\s+(?:or|and)\s+financial\s+advice)',
            r'(personal opinions [^.]*)',
            r'(controversial topics)',
            r'(political ideology)',
            r'(factual accuracy)',
            r'(verifiable evidence)',
            r'(critical thinking [^.]*)',
            r'(bullet points [^.]*)',
            r'(clear headings [^.]*)',
            r'(step-by-step [^.]*)',
            r'(well-researched information [^.]*)',
            r'(clarifying questions [^.]*)',
            r'(multiple perspectives [^.]*)',
            r'(balanced viewpoints [^.]*)',
            r'(appropriate disclaimers [^.]*)',
            r'(professional resources [^.]*)',
        ]

        # stopwords for robust similarity checks
        self._stop = {
            "the","a","an","your","you","and","or","to","of","in","on","for","with","by","as",
            "that","this","these","those","be","is","are","was","were","will","should","must",
            "not","do","does","did","can","could","would","might","may","when","if","unless"
        }

    # ---- public API ----
    def extract_from_labeled_prompts_text(self, txt: str) -> Tuple[List[Dict], Dict]:
        """
        Parse files of the form:
          Prompt 1:\n...text...\n====\nPrompt 2:\n...text...
        Returns (unique_phrases_list, phrase_counts).
        """
        # match each "Prompt N: ... (until divider or EOF)"
        blocks = list(re.finditer(
            r'(?ms)^\s*Prompt\s+(\d+):\s*(.*?)(?:\n=+\s*\n|\Z)',
            txt
        ))

        all_phrases: List[Dict] = []
        counts: Dict[str, int] = defaultdict(int)

        for m in blocks:
            pid = int(m.group(1))
            body = m.group(2).strip()
            if len(body) < 20:
                continue
            phrases = self._extract_one_prompt(body)
            for phrase in phrases:
                item = {
                    "phrase": phrase,
                    "prompt_id": pid,
                    "word_count": len(phrase.split()),
                    "char_length": len(phrase)
                }
                all_phrases.append(item)
                counts[phrase.lower()] += 1

        # cross-prompt unique (keep first occurrence)
        unique, seen = [], set()
        for it in sorted(all_phrases, key=lambda x: (x["prompt_id"], x["phrase"].lower())):
            k = it["phrase"].lower()
            if k not in seen:
                unique.append(it)
                seen.add(k)
        return unique, counts

    # ---- internals ----
    def _extract_one_prompt(self, prompt: str) -> List[str]:
        pattern_groups = [
            self.identity_patterns,
            self.personality_patterns,
            self.behavior_patterns,
            self.constraint_patterns,
            self.technical_patterns,
            self.domain_patterns
        ]

        cands = []
        for group in pattern_groups:
            for pat in group:
                for m in re.finditer(pat, prompt, re.IGNORECASE):
                    text = m.group(1).strip().rstrip('.')
                    if group == self.personality_patterns:
                        text = re.sub(r'^(You are|are)\s+', '', text, flags=re.IGNORECASE)
                    if self._is_valid_phrase(text):
                        cands.append({
                            "text": text,
                            "start": m.start(1),
                            "end":   m.end(1),
                            "tokens": self._norm_tokens(text)
                        })

        if not cands:
            return []

        # 1) suppress fully-contained subspans (keep longest spans)
        cands = self._suppress_subspans(cands)
        # 2) suppress near-duplicates by Jaccard similarity of normalized tokens
        cands = self._suppress_near_duplicates(cands, self.jaccard_thresh)

        # Final order: by first appearance; also dedup by normalized token key
        cands.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        seen_keys, out = set(), []
        for c in cands:
            key = " ".join(c["tokens"])
            if key and key not in seen_keys:
                out.append(c["text"])
                seen_keys.add(key)
        return out

    def _is_valid_phrase(self, phrase: str) -> bool:
        p = phrase.lower().strip()
        if len(phrase) <= 8:
            return False
        if len(phrase.split()) < 2:
            return False
        if p.startswith(('you are', 'you should', 'you must', 'you will')):
            return False
        return True

    def _norm_tokens(self, s: str) -> List[str]:
        toks = re.findall(r"[a-z0-9']+", s.lower())
        return [t for t in toks if t not in self._stop]

    def _suppress_subspans(self, cands: List[Dict]) -> List[Dict]:
        # keep longest spans; drop phrases fully inside another kept span
        cands_sorted = sorted(cands, key=lambda x: (x["end"] - x["start"]), reverse=True)
        kept = []
        for c in cands_sorted:
            if not any(c["start"] >= k["start"] and c["end"] <= k["end"] for k in kept):
                kept.append(c)
        return kept

    def _jaccard(self, a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _suppress_near_duplicates(self, cands: List[Dict], thresh: float) -> List[Dict]:
        # greedy: keep longest-by-token-count first
        cands_sorted = sorted(cands, key=lambda x: (len(x["tokens"]), len(x["text"])), reverse=True)
        kept: List[Dict] = []
        for c in cands_sorted:
            if not any(self._jaccard(c["tokens"], k["tokens"]) >= thresh for k in kept):
                kept.append(c)
        return kept

# ----------------------------------------
# I/O & CLI
# ----------------------------------------

def write_report(phrases: List[Dict], out_path: str) -> None:
    from collections import defaultdict
    wc = defaultdict(int)
    length_cats = {"Short (2-3 words)": 0, "Medium (4-7 words)": 0, "Long (8+ words)": 0}
    
    for it in phrases:
        wc[it["word_count"]] += 1
        if it["word_count"] <= 3:
            length_cats["Short (2-3 words)"] += 1
        elif it["word_count"] <= 7:
            length_cats["Medium (4-7 words)"] += 1
        else:
            length_cats["Long (8+ words)"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CANDIDATE PHRASES FOR KL DIVERGENCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Format: phrase | word_count | source_prompt_id\n")
        f.write("-" * 60 + "\n\n")
        
        # Sort by prompt_id then by phrase for consistent output
        sorted_phrases = sorted(phrases, key=lambda x: (x["prompt_id"], x["phrase"].lower()))
        for it in sorted_phrases:
            f.write(f"\"{it['phrase']}\" | {it['word_count']} words | prompt_{it['prompt_id']}\n")
        
        f.write("\n\nSUMMARY:\n")
        f.write(f"Total unique phrases: {len(phrases)}\n\n")
        f.write("Word count distribution:\n")
        for k in sorted(wc.keys()):
            f.write(f"  {k} words: {wc[k]} phrases\n")
        
        f.write("\nLength categories:\n")
        for cat, count in length_cats.items():
            f.write(f"  {cat}: {count}\n")
        f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Extract non-overlapping phrases from labeled system prompts.")
    ap.add_argument("--input", "-i", default="system_prompts_generated.txt",
                    help="Path to input file in 'Prompt N:' + '====' style.")
    ap.add_argument("--out", "-o", default=None,
                    help="Optional path to write a report (same format).")
    ap.add_argument("--jaccard", type=float, default=0.85,
                    help="Near-duplicate Jaccard threshold (default 0.85).")
    args = ap.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input}")
        return

    extractor = PhraseExtractor(jaccard_thresh=args.jaccard)
    phrases, counts = extractor.extract_from_labeled_prompts_text(txt)

    # ---- print to stdout ----
    print("CANDIDATE PHRASES (NON-OVERLAPPING)")
    print("=" * 60)
    print("Format: phrase | word_count | source_prompt_id")
    print("-" * 60)
    for it in sorted(phrases, key=lambda x: (x["prompt_id"], x["phrase"].lower())):
        print(f"\"{it['phrase']}\" | {it['word_count']} words | prompt_{it['prompt_id']}")
    print("\nSUMMARY:")
    print(f"Total unique phrases: {len(phrases)}")

    if args.out:
        write_report(phrases, args.out)
        print(f"\n[INFO] Written to: {args.out}")

if __name__ == "__main__":
    main()
