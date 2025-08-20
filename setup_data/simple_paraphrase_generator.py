#!/usr/bin/env python3
"""
Simple paraphrase generator using Claude API - no filtering or verification.
"""

import anthropic
import os
import sys

def load_phrases_from_file(filepath: str) -> list:
    """Load phrases from candidate file."""
    phrases = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('"') and '" |' in line:
            phrase = line.split('" |')[0][1:]
            phrases.append(phrase)
    
    print(f"Loaded {len(phrases)} phrases")
    return phrases

def generate_paraphrases(client, text: str, num_candidates: int = 10) -> list:
    """Generate paraphrases using Claude API."""
    try:
        prompt = f"""Generate {num_candidates} paraphrases of the following text. Each paraphrase should preserve the meaning but use different wording.

Original text: "{text}"

Return only the paraphrases, one per line, without numbering or quotes."""

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        paraphrases = []
        for line in response.content[0].text.strip().split('\n'):
            candidate = line.strip()
            if candidate and candidate != text:
                paraphrases.append(candidate)
        
        return paraphrases
    except Exception as e:
        print(f"Error generating paraphrases for '{text}': {e}")
        return []

def save_paraphrases(results: dict, output_file: str):
    """Save generated paraphrases."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CLAUDE-GENERATED PARAPHRASES\n")
        f.write("=" * 30 + "\n\n")
        
        total_variants = 0
        successful_phrases = 0
        
        for i, (original, paraphrases) in enumerate(results.items(), 1):
            f.write(f"{i:3d}. ORIGINAL: \"{original}\"\n")
            
            if paraphrases:
                f.write(f"     PARAPHRASES:\n")
                for j, variant in enumerate(paraphrases, 1):
                    f.write(f"       {j}. \"{variant}\"\n")
                total_variants += len(paraphrases)
                successful_phrases += 1
            else:
                f.write(f"     No paraphrases generated\n")
            
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total phrases: {len(results)}\n")
        f.write(f"  Phrases with paraphrases: {successful_phrases}\n")
        f.write(f"  Total paraphrases: {total_variants}\n")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "candidate_phrases_no_overlap.txt"
    output_file = "simple_paraphrases.txt"
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load phrases
    phrases = load_phrases_from_file(input_file)
    
    # Generate paraphrases
    results = {}
    print(f"Generating paraphrases for {len(phrases)} phrases...")
    
    for i, phrase in enumerate(phrases):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(phrases)} phrases...")
        
        paraphrases = generate_paraphrases(client, phrase, num_candidates=10)
        results[phrase] = paraphrases
    
    # Save results
    save_paraphrases(results, output_file)
    
    # Print summary
    total_phrases = len(results)
    phrases_with_variants = sum(1 for paraphrases in results.values() if paraphrases)
    total_variants = sum(len(paraphrases) for paraphrases in results.values())
    
    print(f"\nPARAPHRASE GENERATION COMPLETE:")
    print(f"  Phrases processed: {total_phrases}")
    print(f"  Phrases with paraphrases: {phrases_with_variants}")
    print(f"  Total paraphrases generated: {total_variants}")
    print(f"  Results saved to: {output_file}")