"""
LLM-based Paraphrase Generator for KL Divergence Analysis
Generates high-quality paraphrases using LLMs (GPT-4, Claude, DeepSeek)
"""

import json
import os
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import requests
from sentence_transformers import SentenceTransformer

@dataclass
class ParaphraseConfig:
    """Configuration for LLM paraphrase generation"""
    provider: str = "together"  # "together", "openai", "anthropic"
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 200
    num_variants_per_call: int = 5
    min_similarity: float = 0.5  # Minimum semantic similarity
    max_similarity: float = 0.9  # Maximum semantic similarity
    
class LLMParaphraseGenerator:
    """Generate paraphrases using LLMs with quality control"""
    
    def __init__(self, config: ParaphraseConfig):
        self.config = config
        self.api_call_count = 0
        self.api_error_count = 0
        
        # Initialize similarity model for quality control
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup API based on provider
        self._setup_api()
        
    def _setup_api(self):
        """Setup API configuration based on provider"""
        if self.config.provider == "together":
            self.config.api_key = self.config.api_key or os.getenv("TOGETHER_API_KEY")
            if not self.config.model_name:
                self.config.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
            self.base_url = "https://api.together.xyz/v1/completions"
            self.headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        elif self.config.provider == "openai":
            self.config.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not self.config.model_name:
                self.config.model_name = "gpt-4-turbo-preview"
        elif self.config.provider == "anthropic":
            self.config.api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.config.model_name:
                self.config.model_name = "claude-3-opus-20240229"
                
        print(f"🚀 LLM Paraphraser initialized: {self.config.provider} - {self.config.model_name}")
    
    def _get_llm_completion(self, prompt: str) -> str:
        """Get completion from LLM with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                
                if self.config.provider == "together":
                    return self._together_completion(prompt)
                elif self.config.provider == "openai":
                    return self._openai_completion(prompt)
                elif self.config.provider == "anthropic":
                    return self._anthropic_completion(prompt)
                    
            except Exception as e:
                self.api_error_count += 1
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
    
    def _together_completion(self, prompt: str) -> str:
        """Get completion from Together.ai"""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "stop": ["---", "\n\n\n"]
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and result['choices']:
            return result['choices'][0]['text'].strip()
        return ""
    
    def _openai_completion(self, prompt: str) -> str:
        """Get completion from OpenAI"""
        import openai
        
        client = openai.OpenAI(api_key=self.config.api_key)
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an expert at generating high-quality paraphrases that maintain semantic meaning while varying expression."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content.strip()
    
    def _anthropic_completion(self, prompt: str) -> str:
        """Get completion from Anthropic"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.config.api_key)
        response = client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    
    def generate_paraphrases(self, phrase: str, num_variants: int = 5, 
                           strategy: str = "balanced") -> List[Tuple[str, float]]:
        """
        Generate high-quality paraphrases using LLM
        
        Args:
            phrase: Original phrase to paraphrase
            num_variants: Number of variants to generate
            strategy: "lexical", "syntactic", "pragmatic", or "balanced"
            
        Returns:
            List of (paraphrase, similarity_score) tuples
        """
        
        # Create strategy-specific prompt
        prompt = self._create_paraphrase_prompt(phrase, num_variants, strategy)
        
        # Get LLM response
        response = self._get_llm_completion(prompt)
        
        # Parse paraphrases from response
        raw_paraphrases = self._parse_paraphrases(response)
        
        # Filter by quality and similarity
        filtered_paraphrases = self._filter_paraphrases(phrase, raw_paraphrases)
        
        # If we need more, make another call
        while len(filtered_paraphrases) < num_variants and len(raw_paraphrases) > 0:
            additional_prompt = self._create_additional_paraphrase_prompt(
                phrase, num_variants - len(filtered_paraphrases), 
                [p[0] for p in filtered_paraphrases]
            )
            additional_response = self._get_llm_completion(additional_prompt)
            additional_paraphrases = self._parse_paraphrases(additional_response)
            
            # Filter new paraphrases
            new_filtered = self._filter_paraphrases(phrase, additional_paraphrases)
            
            # Add unique ones
            for paraphrase, score in new_filtered:
                if not any(p[0].lower() == paraphrase.lower() for p in filtered_paraphrases):
                    filtered_paraphrases.append((paraphrase, score))
        
        return filtered_paraphrases[:num_variants]
    
    def _create_paraphrase_prompt(self, phrase: str, num_variants: int, strategy: str) -> str:
        """Create prompt for paraphrase generation"""
        
        strategy_instructions = {
            "lexical": "Focus on word substitutions and synonyms while keeping the structure similar.",
            "syntactic": "Vary the sentence structure and word order while maintaining meaning.",
            "pragmatic": "Adjust formality, tone, and style while preserving the core message.",
            "balanced": "Use a mix of lexical, syntactic, and pragmatic variations."
        }
        
        prompt = f"""Generate {num_variants} high-quality paraphrases for the following phrase from an AI system prompt:

Original phrase: "{phrase}"

Requirements:
1. Each paraphrase must maintain the exact same meaning
2. {strategy_instructions.get(strategy, strategy_instructions['balanced'])}
3. Suitable for use in AI assistant system prompts
4. Vary in length and style but stay professional
5. Do not use quotation marks in the paraphrases
6. Each paraphrase should be different from the others

Generate exactly {num_variants} paraphrases, one per line:

1. """
        
        return prompt
    
    def _create_additional_paraphrase_prompt(self, phrase: str, num_needed: int, 
                                           existing: List[str]) -> str:
        """Create prompt for additional paraphrases"""
        
        existing_list = "\n".join([f"- {p}" for p in existing[:5]])  # Show max 5
        
        prompt = f"""Generate {num_needed} MORE paraphrases for this phrase:

Original phrase: "{phrase}"

Already generated (avoid these):
{existing_list}

Generate {num_needed} new, different paraphrases that maintain the same meaning:

1. """
        
        return prompt
    
    def _parse_paraphrases(self, response: str) -> List[str]:
        """Parse paraphrases from LLM response"""
        paraphrases = []
        
        # Split by newlines
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Try to extract numbered items (1., 2., etc.)
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                paraphrase = match.group(1).strip()
            else:
                # Try bullet points
                match = re.match(r'^[-•*]\s*(.+)$', line)
                if match:
                    paraphrase = match.group(1).strip()
                else:
                    # Take the whole line if it's not empty
                    paraphrase = line
            
            # Clean up
            paraphrase = paraphrase.strip('"\'')
            paraphrase = re.sub(r'\s+', ' ', paraphrase)
            
            if len(paraphrase) > 5:  # Minimum length check
                paraphrases.append(paraphrase)
        
        return paraphrases
    
    def _filter_paraphrases(self, original: str, paraphrases: List[str]) -> List[Tuple[str, float]]:
        """Filter paraphrases by quality and semantic similarity"""
        
        if not paraphrases:
            return []
        
        # Compute embeddings
        original_embedding = self.similarity_model.encode([original])
        paraphrase_embeddings = self.similarity_model.encode(paraphrases)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(original_embedding, paraphrase_embeddings)[0]
        
        # Filter and sort
        filtered = []
        seen_lower = {original.lower()}
        
        for paraphrase, similarity in zip(paraphrases, similarities):
            # Check if within similarity bounds
            if self.config.min_similarity <= similarity <= self.config.max_similarity:
                # Check if not duplicate
                if paraphrase.lower() not in seen_lower:
                    filtered.append((paraphrase, float(similarity)))
                    seen_lower.add(paraphrase.lower())
        
        # Sort by similarity (descending)
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return filtered
    
    def generate_multi_strategy_paraphrases(self, phrase: str, 
                                          variants_per_strategy: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Generate paraphrases using multiple strategies"""
        
        strategies = ["lexical", "syntactic", "pragmatic"]
        all_paraphrases = {}
        
        for strategy in strategies:
            print(f"  🔄 Generating {strategy} paraphrases...")
            paraphrases = self.generate_paraphrases(
                phrase, variants_per_strategy, strategy
            )
            all_paraphrases[strategy] = paraphrases
        
        return all_paraphrases
    
    def batch_generate_paraphrases(self, phrases: List[str], 
                                 num_variants: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Generate paraphrases for multiple phrases"""
        
        results = {}
        total = len(phrases)
        
        print(f"\n🚀 Generating paraphrases for {total} phrases using {self.config.provider}")
        
        for i, phrase in enumerate(phrases, 1):
            print(f"\n[{i}/{total}] Processing: '{phrase[:50]}...'")
            
            try:
                # Generate paraphrases
                paraphrases = self.generate_paraphrases(phrase, num_variants)
                
                if paraphrases:
                    results[phrase] = paraphrases
                    print(f"  ✅ Generated {len(paraphrases)} paraphrases")
                    
                    # Show examples
                    for j, (para, sim) in enumerate(paraphrases[:3], 1):
                        print(f"     {j}. {para[:60]}... (sim: {sim:.3f})")
                else:
                    print(f"  ❌ No valid paraphrases generated")
                    results[phrase] = []
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[phrase] = []
            
            # Rate limiting
            if i < total:
                time.sleep(0.5)
        
        print(f"\n✅ Completed: {len([r for r in results.values() if r])} successful")
        print(f"🔌 Total API calls: {self.api_call_count}")
        
        return results
    
    def save_results(self, results: Dict[str, List[Tuple[str, float]]], 
                    output_file: str = "llm_generated_paraphrases.json"):
        """Save results in format compatible with KL analysis"""
        
        # Convert to the expected format
        formatted_results = {}
        for phrase, paraphrases in results.items():
            formatted_results[phrase] = {
                "paraphrases": [p[0] for p in paraphrases],
                "similarities": [p[1] for p in paraphrases],
                "count": len(paraphrases)
            }
        
        output_data = {
            "metadata": {
                "generator": "llm",
                "provider": self.config.provider,
                "model": self.config.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "api_calls": self.api_call_count,
                "api_errors": self.api_error_count,
                "total_phrases": len(results),
                "config": {
                    "temperature": self.config.temperature,
                    "min_similarity": self.config.min_similarity,
                    "max_similarity": self.config.max_similarity
                }
            },
            "paraphrases": formatted_results
        }
        
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        # Also save in text format for compatibility
        text_file = output_file.replace('.json', '.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("LLM-GENERATED PARAPHRASES FOR KL DIVERGENCE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Provider: {self.config.provider} | Model: {self.config.model_name}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 60 + "\n\n")
            
            for i, (phrase, paraphrases) in enumerate(results.items(), 1):
                f.write(f"{i}. ORIGINAL: \"{phrase}\"\n")
                if paraphrases:
                    f.write("   SIMILAR PHRASES (LLM-generated):\n")
                    for j, (para, sim) in enumerate(paraphrases, 1):
                        f.write(f"   {j}. \"{para}\" (similarity: {sim:.3f})\n")
                else:
                    f.write("   No paraphrases generated\n")
                f.write("\n")
        
        print(f"\n📁 Results saved:")
        print(f"   - {output_file}")
        print(f"   - {text_file}")

def main():
    """Main function to generate LLM paraphrases"""
    print("🚀 LLM-Based Paraphrase Generator for KL Analysis")
    print("=" * 50)
    
    # Load candidate phrases
    phrases_file = "candidate_phrases_for_kl.txt"
    if not os.path.exists(phrases_file):
        print(f"❌ File not found: {phrases_file}")
        print("Run extract_data.py first to generate candidate phrases")
        return
    
    # Parse phrases from file
    phrases = []
    with open(phrases_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and '"' in line and '|' in line:
                # Extract phrase from format: "phrase" | N words | prompt_X
                match = re.search(r'"([^"]+)"', line)
                if match:
                    phrases.append(match.group(1))
    
    print(f"📊 Loaded {len(phrases)} candidate phrases")
    
    # Select provider
    print("\nSelect LLM provider:")
    print("1. Together.ai (fast, cost-effective)")
    print("2. OpenAI GPT-4 (high quality)")
    print("3. Anthropic Claude (excellent reasoning)")
    
    choice = input("\nEnter choice (1-3) [1]: ").strip() or "1"
    
    provider_map = {
        "1": ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        "2": ("openai", "gpt-4-turbo-preview"),
        "3": ("anthropic", "claude-3-opus-20240229")
    }
    
    if choice not in provider_map:
        print("❌ Invalid choice")
        return
    
    provider, model = provider_map[choice]
    
    # Number of phrases to process
    num_phrases = input(f"\nHow many phrases to process? (1-{len(phrases)}) [10]: ").strip()
    num_phrases = min(int(num_phrases) if num_phrases.isdigit() else 10, len(phrases))
    
    # Initialize generator
    config = ParaphraseConfig(
        provider=provider,
        model_name=model,
        temperature=0.7,
        num_variants_per_call=5,
        min_similarity=0.5,
        max_similarity=0.9
    )
    
    try:
        generator = LLMParaphraseGenerator(config)
        
        # Estimate cost
        estimated_calls = num_phrases * 2  # Assume ~2 calls per phrase
        cost_per_call = {"together": 0.001, "openai": 0.003, "anthropic": 0.004}
        estimated_cost = estimated_calls * cost_per_call.get(provider, 0.002)
        
        print(f"\n💰 Estimated cost: ${estimated_cost:.2f}")
        
        if input("Continue? (y/n): ").lower() != 'y':
            print("❌ Cancelled")
            return
        
        # Generate paraphrases
        selected_phrases = phrases[:num_phrases]
        results = generator.batch_generate_paraphrases(selected_phrases, num_variants=5)
        
        # Save results
        output_file = f"llm_paraphrases_{provider}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        # Summary
        successful = len([r for r in results.values() if r])
        total_paraphrases = sum(len(r) for r in results.values())
        
        print(f"\n🎉 Generation complete!")
        print(f"   Phrases processed: {num_phrases}")
        print(f"   Successful: {successful}")
        print(f"   Total paraphrases: {total_paraphrases}")
        print(f"   API calls: {generator.api_call_count}")
        print(f"   Estimated cost: ${generator.api_call_count * cost_per_call.get(provider, 0.002):.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for API keys
    print("🔑 API Key Setup:")
    print("   Together: export TOGETHER_API_KEY='your-key'")
    print("   OpenAI: export OPENAI_API_KEY='your-key'") 
    print("   Anthropic: export ANTHROPIC_API_KEY='your-key'")
    print()
    
    main()