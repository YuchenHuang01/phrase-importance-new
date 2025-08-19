import numpy as np
from typing import List, Dict, Tuple, Union
import random

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False

class EmbeddingBasedParaphraser:
    """Generate paraphrastic variants using SentenceTransformer embeddings."""
    
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2', use_nli: bool = True):
        if not AVAILABLE:
            raise ImportError("Install: pip install sentence-transformers scikit-learn")
        
        # Use paraphrase-specific model for better semantic similarity
        self.model = SentenceTransformer(model_name)
        self.phrases = []
        self.embeddings = None
        self.similarity_matrix = None
        
        # Optional NLI for entailment filtering
        self.use_nli = use_nli and NLI_AVAILABLE
        if self.use_nli:
            try:
                self.nli_model = pipeline("text-classification", 
                                        model="cross-encoder/nli-MiniLM2-L6-H768",
                                        device=0 if torch.cuda.is_available() else -1)
            except:
                # Fallback to CPU-friendly model
                try:
                    import torch
                    self.nli_model = pipeline("text-classification", 
                                            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                                            device=-1)
                except:
                    print("Warning: NLI model loading failed, continuing without entailment filtering")
                    self.use_nli = False
    
    def load_phrases_from_file(self, filepath: str) -> List[str]:
        """Load phrases from candidate file."""
        phrases = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('"') and '" |' in line:
                phrase = line.split('" |')[0][1:]
                phrases.append(phrase)
        
        print(f"Loaded {len(phrases)} phrases for paraphrase generation")
        self.phrases = phrases
        return phrases
    
    def compute_embeddings(self):
        """Compute embeddings for all phrases."""
        print("Computing phrase embeddings...")
        self.embeddings = self.model.encode(self.phrases, show_progress_bar=True)
        
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        return self.embeddings, self.similarity_matrix
    
    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[bool, float]:
        """Check if hypothesis entails or is entailed by premise using NLI."""
        if not self.use_nli:
            return True, 1.0
        
        try:
            # Check both directions for bidirectional entailment/equivalence
            result1 = self.nli_model(f"{premise} [SEP] {hypothesis}")
            result2 = self.nli_model(f"{hypothesis} [SEP] {premise}")
            
            # Extract entailment scores
            entail_score1 = 0.0
            entail_score2 = 0.0
            
            for res in result1:
                if res['label'] in ['ENTAILMENT', 'entailment']:
                    entail_score1 = res['score']
                    break
            
            for res in result2:
                if res['label'] in ['ENTAILMENT', 'entailment']:
                    entail_score2 = res['score']
                    break
            
            # Consider it valid if either direction has high entailment
            max_entail = max(entail_score1, entail_score2)
            is_entailed = max_entail > 0.5  # Threshold for entailment
            
            return is_entailed, max_entail
        except:
            # Fallback if NLI fails
            return True, 1.0
    
    def find_similar_phrases(self, target_phrase: str, num_variants: int = 10, 
                           min_similarity: float = 0.5, max_similarity: float = 0.85) -> List[Tuple[str, float]]:
        """Find semantically similar phrases to use as paraphrases."""
        if target_phrase not in self.phrases:
            return []
        
        target_idx = self.phrases.index(target_phrase)
        similarities = self.similarity_matrix[target_idx]
        
        # Get phrase-similarity pairs, excluding the target phrase itself
        similar_pairs = []
        for i, similarity in enumerate(similarities):
            if (i != target_idx and 
                min_similarity <= similarity <= max_similarity):
                similar_pairs.append((self.phrases[i], similarity))
        
        # Sort by similarity (descending) and return top variants
        similar_pairs.sort(key=lambda x: x[1], reverse=True)
        return similar_pairs[:num_variants]
    
    def generate_embedding_based_variants(self, phrase: str, num_variants: int = 10) -> List[str]:
        """Generate paraphrastic variants using embedding similarity."""
        similar_phrases = self.find_similar_phrases(phrase, num_variants * 2)  # Get more candidates
        
        variants = []
        seen_similarities = set()
        
        for similar_phrase, similarity in similar_phrases:
            # Tighter bounds: avoid phrases that are too similar (duplicates) or too different
            if 0.5 <= similarity <= 0.85:
                # Check entailment if NLI is enabled
                is_valid = True
                if self.use_nli:
                    is_entailed, entail_score = self.check_entailment(phrase, similar_phrase)
                    is_valid = is_entailed
                
                if is_valid:
                    # Avoid phrases with very similar similarity scores (likely near-duplicates)
                    similarity_rounded = round(similarity, 2)
                    if similarity_rounded not in seen_similarities:
                        variants.append(similar_phrase)
                        seen_similarities.add(similarity_rounded)
        
        return variants[:num_variants]
    
    def generate_semantic_variants(self, phrase: str, num_variants: int = 10) -> List[str]:
        """Generate variants using semantic neighborhood exploration."""
        if phrase not in self.phrases:
            return []
        
        target_idx = self.phrases.index(phrase)
        target_embedding = self.embeddings[target_idx]
        
        # Find phrases in semantic neighborhood
        variants = []
        phrase_scores = []
        
        for i, other_phrase in enumerate(self.phrases):
            if i != target_idx:
                similarity = cosine_similarity([target_embedding], [self.embeddings[i]])[0][0]
                
                # Filter for good paraphrase candidates with tighter bounds
                if 0.5 <= similarity <= 0.85:  # Tighter bounds for better quality
                    # Optional NLI filtering
                    is_valid = True
                    if self.use_nli:
                        is_entailed, entail_score = self.check_entailment(phrase, other_phrase)
                        is_valid = is_entailed
                    
                    if is_valid:
                        phrase_scores.append((other_phrase, similarity, i))
        
        # Sort by similarity and diversify selection
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse variants (avoid clustering around single similarity values)
        selected_similarities = []
        for phrase_text, similarity, idx in phrase_scores:
            # Ensure diversity in similarity scores
            if not any(abs(similarity - prev_sim) < 0.05 for prev_sim in selected_similarities):
                variants.append(phrase_text)
                selected_similarities.append(similarity)
                
                if len(variants) >= num_variants:
                    break
        
        return variants
    
    def generate_contextual_variants(self, phrase: str, num_variants: int = 10) -> List[str]:
        """Generate variants considering phrase context and length."""
        if phrase not in self.phrases:
            return []
        
        target_idx = self.phrases.index(phrase)
        target_length = len(phrase.split())
        
        # Find phrases with similar length and semantic content
        candidates = []
        
        for i, other_phrase in enumerate(self.phrases):
            if i != target_idx:
                other_length = len(other_phrase.split())
                length_diff = abs(target_length - other_length)
                
                # Prefer phrases with similar length (±2 words)
                if length_diff <= 2:
                    similarity = self.similarity_matrix[target_idx, i]
                    if 0.5 <= similarity <= 0.85:  # Tighter bounds
                        # Optional NLI filtering
                        is_valid = True
                        if self.use_nli:
                            is_entailed, entail_score = self.check_entailment(phrase, other_phrase)
                            is_valid = is_entailed
                        
                        if is_valid:
                            # Weight by both similarity and length similarity
                            length_weight = 1.0 / (1.0 + length_diff * 0.1)
                            combined_score = similarity * length_weight
                            candidates.append((other_phrase, combined_score, similarity))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top variants
        return [phrase for phrase, _, _ in candidates[:num_variants]]
    
    def generate_all_paraphrase_variants(self, phrase: str, num_variants: int = 10, 
                                        return_similarities: bool = False) -> Union[List[str], Tuple[List[str], List[float]]]:
        """Generate paraphrastic variants using multiple embedding-based methods."""
        all_variants = []
        
        # Method 1: Direct embedding similarity
        embedding_variants = self.generate_embedding_based_variants(phrase, num_variants // 3)
        all_variants.extend(embedding_variants)
        
        # Method 2: Semantic neighborhood
        semantic_variants = self.generate_semantic_variants(phrase, num_variants // 3)
        all_variants.extend(semantic_variants)
        
        # Method 3: Contextual similarity
        contextual_variants = self.generate_contextual_variants(phrase, num_variants // 3)
        all_variants.extend(contextual_variants)
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = {phrase.lower()}
        
        for variant in all_variants:
            if variant.lower() not in seen:
                unique_variants.append(variant)
                seen.add(variant.lower())
        
        # If we need more variants, get additional ones from direct similarity
        while len(unique_variants) < num_variants:
            additional = self.find_similar_phrases(phrase, num_variants * 2, 
                                                 min_similarity=0.45, max_similarity=0.9)  # Slightly relaxed for fallback
            added_any = False
            
            for similar_phrase, _ in additional:
                if similar_phrase.lower() not in seen and len(unique_variants) < num_variants:
                    unique_variants.append(similar_phrase)
                    seen.add(similar_phrase.lower())
                    added_any = True
            
            if not added_any:  # Prevent infinite loop
                break
        
        final_variants = unique_variants[:num_variants]
        
        if return_similarities:
            # Calculate similarities for final variants
            similarities = []
            if phrase in self.phrases:
                phrase_idx = self.phrases.index(phrase)
                for variant in final_variants:
                    if variant in self.phrases:
                        variant_idx = self.phrases.index(variant)
                        sim = self.similarity_matrix[phrase_idx, variant_idx]
                    else:
                        # Compute similarity on the fly
                        phrase_emb = self.embeddings[phrase_idx]
                        variant_emb = self.model.encode([variant])[0]
                        sim = cosine_similarity([phrase_emb], [variant_emb])[0][0]
                    similarities.append(sim)
            return final_variants, similarities
        
        return final_variants
    
    def generate_paraphrases_for_all_phrases(self, num_variants: int = 10) -> Dict[str, Dict]:
        """Generate paraphrases for all phrases with metadata."""
        if self.embeddings is None:
            self.compute_embeddings()
        
        results = {}
        
        print(f"Generating {num_variants} embedding-based paraphrases for {len(self.phrases)} phrases...")
        
        for i, phrase in enumerate(self.phrases):
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(self.phrases)} phrases...")
            
            variants = self.generate_all_paraphrase_variants(phrase, num_variants)
            
            # Calculate similarity scores for the variants
            variant_similarities = []
            if variants:
                phrase_idx = self.phrases.index(phrase)
                for variant in variants:
                    if variant in self.phrases:
                        variant_idx = self.phrases.index(variant)
                        similarity = self.similarity_matrix[phrase_idx, variant_idx]
                        variant_similarities.append(similarity)
                    else:
                        variant_similarities.append(0.0)
            
            results[phrase] = {
                'variants': variants,
                'similarities': variant_similarities,
                'num_variants_found': len(variants)
            }
        
        return results

def save_embedding_paraphrases(results: Dict[str, Dict], output_file: str):
    """Save embedding-based paraphrases for KL analysis."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EMBEDDING-BASED PARAPHRASES FOR KL DIVERGENCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Generated using SentenceTransformer semantic similarity\n")
        f.write("Format: Original phrase B -> Similar phrases {B_1, B_2, ..., B_k}\n")
        f.write("-" * 60 + "\n\n")
        
        total_variants = 0
        successful_phrases = 0
        
        for i, (original, data) in enumerate(results.items(), 1):
            variants = data['variants']
            similarities = data['similarities']
            
            f.write(f"{i:3d}. ORIGINAL: \"{original}\"\n")
            
            if variants:
                f.write(f"     SIMILAR PHRASES (semantic variants):\n")
                for j, (variant, similarity) in enumerate(zip(variants, similarities), 1):
                    f.write(f"       {j}. \"{variant}\" (similarity: {similarity:.3f})\n")
                total_variants += len(variants)
                successful_phrases += 1
            else:
                f.write(f"     No suitable semantic variants found\n")
            
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total original phrases: {len(results)}\n")
        f.write(f"  Phrases with variants: {successful_phrases}\n")
        f.write(f"  Total variants generated: {total_variants}\n")
        if successful_phrases > 0:
            f.write(f"  Average variants per successful phrase: {total_variants/successful_phrases:.1f}\n")
        f.write(f"\nFor KL analysis: Replace each B with semantically similar B_i\n")
        f.write(f"and measure KL divergence between output distributions.\n")

def analyze_paraphrase_quality(results: Dict[str, Dict]):
    """Analyze the quality of generated paraphrases."""
    total_phrases = len(results)
    phrases_with_variants = sum(1 for data in results.values() if data['variants'])
    total_variants = sum(len(data['variants']) for data in results.values())
    
    # Similarity distribution
    all_similarities = []
    for data in results.values():
        all_similarities.extend(data['similarities'])
    
    if all_similarities:
        mean_similarity = np.mean(all_similarities)
        std_similarity = np.std(all_similarities)
        
        print(f"\nPARAPHRASE QUALITY ANALYSIS:")
        print(f"  Phrases processed: {total_phrases}")
        print(f"  Phrases with variants: {phrases_with_variants} ({phrases_with_variants/total_phrases*100:.1f}%)")
        print(f"  Total variants generated: {total_variants}")
        print(f"  Mean similarity score: {mean_similarity:.3f}")
        print(f"  Similarity std dev: {std_similarity:.3f}")
        print(f"  Similarity range: {min(all_similarities):.3f} - {max(all_similarities):.3f}")

def main():
    """Generate embedding-based paraphrases for KL divergence analysis."""
    if not AVAILABLE:
        print("Error: Required packages not installed")
        print("Run: pip install sentence-transformers scikit-learn transformers")
        return
    
    print("Embedding-Based Paraphrase Generator for KL Analysis")
    print("=" * 55)
    print("Using paraphrase-specific model with tighter similarity bounds (0.5-0.85)")
    if NLI_AVAILABLE:
        print("NLI entailment filtering enabled")
    else:
        print("NLI entailment filtering disabled (transformers not available)")
    print()
    
    try:
        paraphraser = EmbeddingBasedParaphraser(
            model_name='paraphrase-MiniLM-L6-v2',
            use_nli=True
        )
        
        # Load phrases from your file
        phrases = paraphraser.load_phrases_from_file('candidate_phrases_no_overlap.txt')
        
        # Compute embeddings and similarity matrix
        paraphraser.compute_embeddings()
        
        # Generate embedding-based paraphrases
        results = paraphraser.generate_paraphrases_for_all_phrases(num_variants=10)
        
        # Analyze quality
        analyze_paraphrase_quality(results)
        
        # Save results
        save_embedding_paraphrases(results, 'embedding_based_paraphrases.txt')
        
        # Show examples
        print(f"\nExample embedding-based paraphrases:")
        example_phrases = list(results.items())[:3]
        for original, data in example_phrases:
            variants = data['variants']
            similarities = data['similarities']
            
            print(f"\nOriginal: \"{original}\"")
            if variants:
                for i, (variant, sim) in enumerate(zip(variants[:5], similarities[:5]), 1):
                    print(f"  {i}. \"{variant}\" (sim: {sim:.3f})")
            else:
                print(f"  No variants found")
        
        print(f"\nResults saved to: embedding_based_paraphrases.txt")
        print(f"\nReady for KL divergence analysis!")
        print(f"Each phrase B now has semantically similar variants from your phrase set.")
        
    except FileNotFoundError:
        print("Error: candidate_phrases_no_overlap.txt not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()