"""
User Prompts Generator for KL Divergence Analysis
File 1/4: Generate standardized user prompts for phrase importance testing
"""

from typing import List, Dict
import json
import random

class UserPromptGenerator:
    """Generate and manage user prompts for phrase importance analysis."""
    
    def __init__(self):
        self.prompts_by_category = self._create_categorized_prompts()
        self.all_prompts = self._flatten_prompts()
    
    def _create_categorized_prompts(self) -> Dict[str, List[str]]:
        """Create prompts organized by category for analysis."""
        return {
            "general_knowledge": [
                "What is artificial intelligence?",
                "Explain how machine learning works",
                "What are the main causes of climate change?",
                "How does the human brain process information?",
                "Explain quantum computing in simple terms"
            ],
            
            "problem_solving": [
                "I'm feeling overwhelmed with work. What should I do?",
                "How can I improve my communication skills?",
                "What's the best way to learn a new programming language?",
                "I need help planning a budget. Where do I start?",
                "How do I deal with conflict in a team setting?"
            ],
            
            "creative_open_ended": [
                "Write a short story about time travel",
                "What would happen if gravity suddenly stopped working?",
                "Describe the perfect day from your perspective",
                "Create a recipe for happiness",
                "If you could redesign the internet, how would you do it?"
            ],
            
            "analytical_reasoning": [
                "Compare the pros and cons of renewable energy",
                "Analyze the impact of social media on society",
                "What are the ethical implications of genetic engineering?",
                "Explain the relationship between inflation and unemployment",
                "Evaluate the benefits and risks of space exploration"
            ]
        }
    
    def _flatten_prompts(self) -> List[str]:
        """Flatten categorized prompts into single list."""
        all_prompts = []
        for category_prompts in self.prompts_by_category.values():
            all_prompts.extend(category_prompts)
        return all_prompts
    
    def get_all_prompts(self) -> List[str]:
        """Get all 20 user prompts as a single list."""
        return self.all_prompts.copy()
    
    def get_prompts_by_category(self) -> Dict[str, List[str]]:
        """Get prompts organized by category."""
        return self.prompts_by_category.copy()
    
    def get_category_prompts(self, category: str) -> List[str]:
        """Get prompts from specific category."""
        return self.prompts_by_category.get(category, [])
    
    def get_random_subset(self, n: int, seed: int = 42) -> List[str]:
        """Get random subset of n prompts with reproducible seed."""
        random.seed(seed)
        return random.sample(self.all_prompts, min(n, len(self.all_prompts)))
    
    def validate_prompts(self) -> Dict[str, any]:
        """Validate prompt set and return statistics."""
        stats = {
            "total_prompts": len(self.all_prompts),
            "categories": len(self.prompts_by_category),
            "prompts_per_category": {
                cat: len(prompts) for cat, prompts in self.prompts_by_category.items()
            },
            "avg_prompt_length": sum(len(p.split()) for p in self.all_prompts) / len(self.all_prompts),
            "unique_prompts": len(set(self.all_prompts)) == len(self.all_prompts)
        }
        return stats
    
    def save_prompts_to_file(self, filepath: str = "user_prompts.json"):
        """Save prompts to JSON file for reproducibility."""
        data = {
            "metadata": {
                "total_prompts": len(self.all_prompts),
                "categories": list(self.prompts_by_category.keys()),
                "description": "Standardized user prompts for KL divergence phrase importance analysis"
            },
            "prompts_by_category": self.prompts_by_category,
            "all_prompts": self.all_prompts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"User prompts saved to: {filepath}")
        return filepath
    
    def load_prompts_from_file(self, filepath: str = "user_prompts.json"):
        """Load prompts from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.prompts_by_category = data["prompts_by_category"]
        self.all_prompts = data["all_prompts"]
        
        print(f"User prompts loaded from: {filepath}")
        return data["metadata"]

def create_system_prompt_templates() -> Dict[str, str]:
    """Create different system prompt templates for testing."""
    return {
        "claude_style": "You are Claude, an AI assistant created by Anthropic. You are [PHRASE]. Please be helpful and informative.",
        
        "simple": "You are an AI assistant. You are [PHRASE]. Please help the user.",
        
        "detailed": "You are an AI assistant designed to help users. You are [PHRASE]. Please provide accurate, helpful, and well-reasoned responses to user questions.",
        
        "conversational": "You are a helpful AI assistant. You are [PHRASE]. Please engage with the user in a friendly and informative manner.",
        
        "professional": "You are a professional AI assistant. You are [PHRASE]. Please provide clear, accurate, and actionable information to assist users."
    }

def main():
    """Main function to generate and save user prompts."""
    print("User Prompts Generator for KL Divergence Analysis")
    print("=" * 55)
    
    # Create prompt generator
    generator = UserPromptGenerator()
    
    # Validate prompts
    stats = generator.validate_prompts()
    print(f"\nPrompt Statistics:")
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Average length: {stats['avg_prompt_length']:.1f} words")
    print(f"  All unique: {stats['unique_prompts']}")
    
    # Show prompts by category
    print(f"\nPrompts by Category:")
    for category, prompts in generator.get_prompts_by_category().items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(prompts)} prompts):")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")
    
    # Save to file
    generator.save_prompts_to_file("user_prompts.json")
    
    # Create and save system prompt templates
    templates = create_system_prompt_templates()
    with open("system_prompt_templates.json", 'w', encoding='utf-8') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\nSystem prompt templates saved to: system_prompt_templates.json")
    
    print(f"\nFiles created:")
    print(f"  - user_prompts.json (20 standardized user prompts)")
    print(f"  - system_prompt_templates.json (5 system prompt templates)")
    
    print(f"\nReady for KL divergence analysis!")
    print(f"These prompts will be used consistently across all phrase importance measurements.")

if __name__ == "__main__":
    main()