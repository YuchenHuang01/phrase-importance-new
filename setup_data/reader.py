#!/usr/bin/env python3
"""
File Structure Reader - Reads and analyzes your existing data files
"""

import os
import re
import json
from typing import Dict, List

def read_extracted_prompts(filepath: str) -> List[str]:
    """Read extracted system prompts from your file."""
    print(f"📖 Reading extracted prompts from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    prompts = []
    sections = re.split(r'Prompt \d+:', content)
    
    for section in sections[1:]:  # Skip header
        lines = section.strip().split('\n')
        prompt_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('='):
                prompt_lines.append(line)
        
        if prompt_lines:
            prompt = ' '.join(prompt_lines)
            prompts.append(prompt)
    
    return prompts

def read_embedding_paraphrases(filepath: str) -> Dict[str, List[str]]:
    """Read embedding-based paraphrases from your file."""
    print(f"📖 Reading embedding paraphrases from: {filepath}")
    
    phrases_dict = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by numbered sections
    sections = re.split(r'\n\s*\d+\.\s+ORIGINAL:', content)
    
    for section in sections[1:]:
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        # Extract original phrase
        first_line = lines[0].strip()
        original_match = re.search(r'"([^"]+)"', first_line)
        if not original_match:
            continue
        
        original = original_match.group(1)
        
        # Extract variants
        variants = []
        in_similar_section = False
        
        for line in lines[1:]:
            line = line.strip()
            if 'SIMILAR PHRASES' in line:
                in_similar_section = True
                continue
            
            if in_similar_section and line:
                # Look for numbered variants with quotes
                variant_match = re.search(r'\d+\.\s+"([^"]+)"', line)
                if variant_match:
                    variant = variant_match.group(1)
                    variants.append(variant)
                # Also try similarity lines
                elif '"' in line and 'similarity:' in line:
                    variant_match = re.search(r'"([^"]+)"\s*\(similarity:', line)
                    if variant_match:
                        variants.append(variant_match.group(1))
        
        if variants:
            phrases_dict[original] = variants[:5]  # Top 5 variants
    
    return phrases_dict

def read_candidate_phrases(filepath: str) -> List[Dict]:
    """Read candidate phrases from your file."""
    print(f"📖 Reading candidate phrases from: {filepath}")
    
    phrases = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('-') and '|' in line:
                # Format: "phrase" | N words | prompt_X
                parts = line.split('|')
                if len(parts) >= 3:
                    phrase = parts[0].strip().strip('"')
                    word_count = parts[1].strip()
                    prompt_id = parts[2].strip()
                    phrases.append({
                        'phrase': phrase,
                        'word_count': word_count,
                        'prompt_id': prompt_id
                    })
    
    return phrases

def analyze_file_structure():
    """Analyze your existing file structure."""
    print("🔍 Analyzing File Structure")
    print("=" * 30)
    
    # Check for extracted prompts
    extracted_paths = [
        'setup_data/extracted_prompts.txt',
        'extracted_prompts.txt'
    ]
    
    extracted_prompts = None
    for path in extracted_paths:
        if os.path.exists(path):
            try:
                extracted_prompts = read_extracted_prompts(path)
                print(f"✅ Found {len(extracted_prompts)} extracted prompts in {path}")
                
                # Show sample
                if extracted_prompts:
                    print(f"📝 Sample prompt: {extracted_prompts[0][:100]}...")
                break
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
    
    if not extracted_prompts:
        print("❌ No extracted prompts found")
    
    print()
    
    # Check for embedding paraphrases
    embedding_paths = [
        'setup_data/embedding_based_paraphrases.txt',
        'embedding_based_paraphrases.txt'
    ]
    
    embedding_phrases = None
    for path in embedding_paths:
        if os.path.exists(path):
            try:
                embedding_phrases = read_embedding_paraphrases(path)
                print(f"✅ Found {len(embedding_phrases)} phrases with paraphrases in {path}")
                
                # Show sample
                if embedding_phrases:
                    first_phrase = list(embedding_phrases.keys())[0]
                    variants = embedding_phrases[first_phrase]
                    print(f"📝 Sample: '{first_phrase}' has {len(variants)} variants")
                break
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
    
    if not embedding_phrases:
        print("❌ No embedding paraphrases found")
    
    print()
    
    # Check for candidate phrases
    candidate_paths = [
        'setup_data/candidate_phrases_for_kl.txt',
        'candidate_phrases_for_kl.txt'
    ]
    
    candidate_phrases = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                candidate_phrases = read_candidate_phrases(path)
                print(f"✅ Found {len(candidate_phrases)} candidate phrases in {path}")
                
                # Show sample
                if candidate_phrases:
                    sample = candidate_phrases[0]
                    print(f"📝 Sample: '{sample['phrase'][:50]}...' ({sample['word_count']} from {sample['prompt_id']})")
                break
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
    
    if not candidate_phrases:
        print("❌ No candidate phrases found")
    
    print()
    
    # Summary
    print("📊 Summary:")
    print(f"  Extracted prompts: {'✅' if extracted_prompts else '❌'} ({len(extracted_prompts) if extracted_prompts else 0})")
    print(f"  Embedding phrases: {'✅' if embedding_phrases else '❌'} ({len(embedding_phrases) if embedding_phrases else 0})")
    print(f"  Candidate phrases: {'✅' if candidate_phrases else '❌'} ({len(candidate_phrases) if candidate_phrases else 0})")
    
    if extracted_prompts and embedding_phrases:
        print("\n🚀 Ready for KL analysis!")
        return True
    else:
        print("\n❌ Missing required files for KL analysis")
        return False

def main():
    """Main function"""
    print("📁 File Structure Analysis for KL Divergence")
    print("=" * 45)
    
    ready = analyze_file_structure()
    
    if ready:
        print("\n✅ All files found and readable!")
        print("\n📋 Next steps:")
        print("  1. Set API key: export TOGETHER_API_KEY='your-key'")
        print("  2. Run: python api_kl_computer.py")
    else:
        print("\n🛠️  Required files:")
        print("  - setup_data/extracted_prompts.txt")
        print("  - setup_data/embedding_based_paraphrases.txt")
        print("  - setup_data/candidate_phrases_for_kl.txt (optional)")

if __name__ == "__main__":
    main()