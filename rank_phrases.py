#!/usr/bin/env python3
"""
Rank phrases by their KL divergence scores within each system prompt.

This script:
1. Loads KL divergence results from compute_kl_divergence.py
2. For each system prompt, shows the ranking of phrases by importance
3. Generates visualizations and reports for each prompt
4. Identifies the most important phrases per system prompt
"""

import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rank_phrase_importance.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PhraseImportanceRanker:
    def __init__(self, results_file: str):
        """Initialize the ranker with results from KL divergence computation."""
        self.results_file = results_file
        self.results_by_prompt = {}
        
    def load_results(self):
        """Load KL divergence results."""
        logger.info(f"Loading results from {self.results_file}")
        
        try:
            with open(self.results_file, 'rb') as f:
                self.results_by_prompt = pickle.load(f)
            logger.info(f"Loaded results for {len(self.results_by_prompt)} system prompts")
        except FileNotFoundError:
            logger.error(f"Results file not found: {self.results_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def generate_reports(self, output_dir: str = 'results'):
        """Generate comprehensive reports for each system prompt."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate reports for each prompt
        for prompt_id, prompt_data in self.results_by_prompt.items():
            self._generate_prompt_report(prompt_id, prompt_data, output_dir, timestamp)
        
        # Generate overall summary
        self._generate_overall_summary(output_dir, timestamp)
        
        # Generate final visualization
        self._create_final_visualization(output_dir, timestamp)
        
        logger.info(f"Reports generated in {output_dir}")
    
    def _generate_prompt_report(self, prompt_id: str, prompt_data: Dict, output_dir: str, timestamp: str):
        """Generate report for a single system prompt."""
        # Only generate text report - no CSV or visualizations needed
        prompt_dir = os.path.join(output_dir, prompt_id)
        os.makedirs(prompt_dir, exist_ok=True)
        
        self._generate_prompt_text_report(prompt_id, prompt_data, 
                                         os.path.join(prompt_dir, f'report_{timestamp}.txt'))
    
    def _create_final_visualization(self, output_dir: str, timestamp: str):
        """Create final summary visualization across all prompts."""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Collect top phrase from each prompt
        top_phrases_data = []
        for prompt_id, prompt_data in self.results_by_prompt.items():
            if not prompt_data['phrase_results']:
                continue
            
            # Find top phrase in this prompt
            top_phrase, top_data = max(
                prompt_data['phrase_results'].items(),
                key=lambda x: x[1]['avg_kl_score']
            )
            
            top_phrases_data.append({
                'prompt_id': prompt_id,
                'phrase': top_phrase[:50] + '...' if len(top_phrase) > 50 else top_phrase,
                'score': top_data['avg_kl_score']
            })
        
        # Sort by score
        top_phrases_data.sort(key=lambda x: x['score'], reverse=True)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_phrases_data) * 0.6)))
        
        y_labels = [f"{item['prompt_id']}: {item['phrase']}" for item in top_phrases_data]
        scores = [item['score'] for item in top_phrases_data]
        
        bars = ax.barh(y_labels, scores)
        ax.set_xlabel('KL Divergence Score')
        ax.set_title('Most Important Phrase per System Prompt')
        ax.invert_yaxis()
        
        # Color bars by score
        if len(scores) > 1:
            norm = plt.Normalize(min(scores), max(scores))
            colors = plt.cm.viridis(norm(scores))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'final_phrase_importance_summary_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Final visualization saved: final_phrase_importance_summary_{timestamp}.png")
    
    def _generate_prompt_text_report(self, prompt_id: str, prompt_data: Dict, filepath: str):
        """Generate detailed text report for a single prompt."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"PHRASE IMPORTANCE ANALYSIS - {prompt_id.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System Prompt ID: {prompt_id}\n")
            f.write(f"Total phrases analyzed: {len(prompt_data['phrase_results'])}\n\n")
            
            # Show the system prompt
            f.write("SYSTEM PROMPT TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{prompt_data['prompt_text']}\n\n")
            
            # Summary statistics
            scores = [data['avg_kl_score'] for data in prompt_data['phrase_results'].values()]
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean KL score: {np.mean(scores):.6f}\n")
            f.write(f"Median KL score: {np.median(scores):.6f}\n")
            f.write(f"Std deviation: {np.std(scores):.6f}\n")
            f.write(f"Min score: {np.min(scores):.6f}\n")
            f.write(f"Max score: {np.max(scores):.6f}\n\n")
            
            # Rankings
            sorted_phrases = sorted(
                prompt_data['phrase_results'].items(),
                key=lambda x: x[1]['avg_kl_score'],
                reverse=True
            )
            
            f.write("PHRASE RANKINGS (by importance)\n")
            f.write("-" * 40 + "\n")
            for i, (phrase, data) in enumerate(sorted_phrases, 1):
                f.write(f"\n{i}. \"{phrase}\"\n")
                f.write(f"   KL Score: {data['avg_kl_score']:.6f}\n")
                f.write(f"   Paraphrases tested: {data['num_paraphrases']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
        
        logger.info(f"Text report saved for {prompt_id}: {filepath}")
    
    def _generate_overall_summary(self, output_dir: str, timestamp: str):
        """Generate overall summary across all prompts."""
        summary_file = os.path.join(output_dir, f'overall_summary_{timestamp}.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("OVERALL PHRASE IMPORTANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total system prompts analyzed: {len(self.results_by_prompt)}\n\n")
            
            # For each prompt, show top phrase
            f.write("MOST IMPORTANT PHRASE PER SYSTEM PROMPT\n")
            f.write("-" * 50 + "\n")
            
            overall_top_phrases = []
            
            for prompt_id, prompt_data in self.results_by_prompt.items():
                if not prompt_data['phrase_results']:
                    continue
                
                # Find top phrase
                top_phrase, top_data = max(
                    prompt_data['phrase_results'].items(),
                    key=lambda x: x[1]['avg_kl_score']
                )
                
                f.write(f"\n{prompt_id}:\n")
                f.write(f"  Top phrase: \"{top_phrase}\"\n")
                f.write(f"  KL Score: {top_data['avg_kl_score']:.6f}\n")
                f.write(f"  Total phrases in prompt: {len(prompt_data['phrase_results'])}\n")
                
                overall_top_phrases.append({
                    'prompt_id': prompt_id,
                    'phrase': top_phrase,
                    'score': top_data['avg_kl_score']
                })
            
            # Overall top phrases across all prompts
            f.write(f"\n\nTOP 10 MOST IMPORTANT PHRASES ACROSS ALL PROMPTS\n")
            f.write("-" * 50 + "\n")
            
            overall_top_phrases.sort(key=lambda x: x['score'], reverse=True)
            for i, item in enumerate(overall_top_phrases[:10], 1):
                f.write(f"\n{i}. \"{item['phrase']}\" (from {item['prompt_id']})\n")
                f.write(f"   KL Score: {item['score']:.6f}\n")
            
            # Most common important phrases
            phrase_counts = {}
            for prompt_data in self.results_by_prompt.values():
                for phrase in prompt_data['phrase_results'].keys():
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
            common_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"\n\nMOST FREQUENTLY APPEARING PHRASES\n")
            f.write("-" * 40 + "\n")
            for phrase, count in common_phrases[:10]:
                f.write(f"\n\"{phrase}\" - appears in {count} prompts\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
        
        # Also save as JSON
        json_summary = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'total_prompts': len(self.results_by_prompt)
            },
            'top_phrases_per_prompt': [
                {
                    'prompt_id': item['prompt_id'],
                    'phrase': item['phrase'],
                    'kl_score': float(item['score'])
                }
                for item in overall_top_phrases
            ],
            'most_common_phrases': [
                {
                    'phrase': phrase,
                    'frequency': count
                }
                for phrase, count in common_phrases[:20]
            ]
        }
        
        json_file = os.path.join(output_dir, f'overall_summary_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Overall summary saved: {summary_file} and {json_file}")
    
    def get_top_phrase_per_prompt(self) -> Dict[str, Dict]:
        """Return the most important phrase for each system prompt."""
        top_phrases = {}
        
        for prompt_id, prompt_data in self.results_by_prompt.items():
            if not prompt_data['phrase_results']:
                continue
            
            top_phrase, top_data = max(
                prompt_data['phrase_results'].items(),
                key=lambda x: x[1]['avg_kl_score']
            )
            
            top_phrases[prompt_id] = {
                'phrase': top_phrase,
                'kl_score': top_data['avg_kl_score'],
                'num_paraphrases': top_data['num_paraphrases']
            }
        
        return top_phrases


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rank phrases by importance per system prompt")
    parser.add_argument('--results', default='results/kl_divergence_results_per_prompt.pkl',
                        help='Path to KL divergence results file')
    parser.add_argument('--output', default='results',
                        help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize ranker
    ranker = PhraseImportanceRanker(args.results)
    
    # Load results
    ranker.load_results()
    
    # Generate reports
    ranker.generate_reports(args.output)
    
    # Print summary
    top_phrases = ranker.get_top_phrase_per_prompt()
    if top_phrases:
        print("\n" + "=" * 80)
        print("MOST IMPORTANT PHRASE PER SYSTEM PROMPT:")
        print("=" * 80)
        for prompt_id, data in top_phrases.items():
            print(f"\n{prompt_id}:")
            print(f"  Phrase: \"{data['phrase']}\"")
            print(f"  KL Score: {data['kl_score']:.6f}")
        print("=" * 80 + "\n")
    
    logger.info("Ranking complete!")


if __name__ == "__main__":
    main()