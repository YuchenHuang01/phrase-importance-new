import re
from typing import List, Dict, Tuple
from collections import defaultdict

class PhraseExtractor:
    """Extract multi-word phrases from system prompts for importance analysis."""
    
    def __init__(self):
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
    
    def extract_phrases_from_prompt(self, prompt: str) -> List[str]:
        """Extract meaningful multi-word phrases from a single prompt."""
        clean_prompt = re.sub(r'={3,}', '', prompt).strip()
        if len(clean_prompt) < 20:
            return []
        
        phrases = []
        
        pattern_groups = [
            self.identity_patterns,
            self.personality_patterns,
            self.behavior_patterns,
            self.constraint_patterns,
            self.technical_patterns,
            self.domain_patterns
        ]
        
        for patterns in pattern_groups:
            for pattern in patterns:
                matches = re.finditer(pattern, clean_prompt, re.IGNORECASE)
                for match in matches:
                    phrase = match.group(1).strip().rstrip('.')
                    
                    if patterns == self.personality_patterns:
                        phrase = re.sub(r'^You are\s+', '', phrase, flags=re.IGNORECASE)
                        phrase = re.sub(r'^are\s+', '', phrase, flags=re.IGNORECASE)
                    
                    if self._is_valid_phrase(phrase):
                        phrases.append(phrase)
        
        return self._deduplicate_phrases(phrases)
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if phrase meets quality criteria."""
        phrase_lower = phrase.lower().strip()
        
        if (len(phrase) <= 8 or 
            len(phrase.split()) < 2 or
            phrase_lower.startswith(('you are', 'you should', 'you must', 'you will'))):
            return False
        
        return True
    
    def _deduplicate_phrases(self, phrases: List[str]) -> List[str]:
        """Remove duplicate phrases while preserving order."""
        unique_phrases = []
        seen = set()
        
        for phrase in phrases:
            phrase_clean = re.sub(r'\s+', ' ', phrase.strip())
            phrase_key = phrase_clean.lower()
            
            if phrase_key not in seen:
                unique_phrases.append(phrase_clean)
                seen.add(phrase_key)
        
        return unique_phrases
    
    def extract_from_file(self, filepath: str) -> Tuple[List[Dict], Dict]:
        """Extract phrases from extracted_prompts.txt file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            prompt_sections = re.split(r'Prompt \d+:\s*', content)
            prompt_sections = [section.strip() for section in prompt_sections if section.strip()]
            
            prompts = []
            for section in prompt_sections:
                clean_section = re.sub(r'={10,}', '', section).strip()
                if len(clean_section) > 20:
                    prompts.append(clean_section)
            
            all_phrases = []
            phrase_counts = defaultdict(int)
            
            for i, prompt in enumerate(prompts, 1):
                phrases = self.extract_phrases_from_prompt(prompt)
                
                for phrase in phrases:
                    all_phrases.append({
                        'phrase': phrase,
                        'prompt_id': i,
                        'word_count': len(phrase.split()),
                        'char_length': len(phrase)
                    })
                    phrase_counts[phrase.lower()] += 1
            
            unique_phrases = []
            seen_phrases = set()
            
            for item in all_phrases:
                phrase_key = item['phrase'].lower()
                if phrase_key not in seen_phrases:
                    unique_phrases.append(item)
                    seen_phrases.add(phrase_key)
            
            return unique_phrases, phrase_counts
            
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return [], {}
    
    def save_phrases(self, phrases: List[Dict], output_file: str) -> None:
        """Save phrases for KL divergence analysis."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CANDIDATE PHRASES FOR KL DIVERGENCE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write("Format: phrase | word_count | source_prompt_id\n")
            f.write("-" * 60 + "\n\n")
            
            for item in phrases:
                f.write(f'"{item["phrase"]}" | {item["word_count"]} words | prompt_{item["prompt_id"]}\n')
            
            f.write(f"\n\nSUMMARY:\n")
            f.write(f"Total unique phrases: {len(phrases)}\n")
            
            word_counts = defaultdict(int)
            for item in phrases:
                word_counts[item['word_count']] += 1
            
            f.write(f"\nWord count distribution:\n")
            for wc in sorted(word_counts.keys()):
                f.write(f"  {wc} words: {word_counts[wc]} phrases\n")
            
            short_phrases = [p for p in phrases if p['word_count'] <= 3]
            medium_phrases = [p for p in phrases if 4 <= p['word_count'] <= 7]
            long_phrases = [p for p in phrases if p['word_count'] >= 8]
            
            f.write(f"\nLength categories:\n")
            f.write(f"  Short (2-3 words): {len(short_phrases)}\n")
            f.write(f"  Medium (4-7 words): {len(medium_phrases)}\n")
            f.write(f"  Long (8+ words): {len(long_phrases)}\n")

def main():
    """Main extraction workflow."""
    print("Extracting phrases from extracted_prompts.txt")
    print("=" * 50)
    
    extractor = PhraseExtractor()
    phrases, phrase_counts = extractor.extract_from_file('extracted_prompts.txt')
    
    if phrases:
        extractor.save_phrases(phrases, 'candidate_phrases_for_kl.txt')
        
        print(f"Processed phrases: {len(phrases)}")
        
        word_count_dist = defaultdict(int)
        for item in phrases:
            word_count_dist[item['word_count']] += 1
        
        print(f"\nWord count distribution:")
        for wc in sorted(word_count_dist.keys())[:10]:
            print(f"  {wc} words: {word_count_dist[wc]} phrases")
        
        print(f"\nExample phrases:")
        for i, item in enumerate(phrases[:10], 1):
            print(f'  {i}. "{item["phrase"]}" ({item["word_count"]} words)')
        
        if len(phrases) > 10:
            print(f"  ... and {len(phrases) - 10} more")
        
        print(f"\nOutput saved to: candidate_phrases_for_kl.txt")
        print(f"Ready for KL divergence analysis with {len(phrases)} candidate phrases")
        
    else:
        print("No phrases extracted. Check input file.")

if __name__ == "__main__":
    main()