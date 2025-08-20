#!/usr/bin/env python3
"""
Download SystemCheck dataset and convert to system_prompts_generated.txt format.
"""

import json
import requests
from datasets import load_dataset

def download_systemcheck():
    """Download and format SystemCheck dataset."""
    
    print("Loading SystemCheck dataset from HuggingFace...")
    
    try:
        # Load the dataset - use 'prompts' config for system prompts
        dataset = load_dataset("normster/SystemCheck", "prompts")
        
        # Get the data - usually in 'train' split
        data = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        
        print(f"Loaded {len(data)} entries from SystemCheck dataset")
        
        # Convert to the same format as system_prompts_generated.txt
        output_lines = []
        
        for i, item in enumerate(data, 1):
            # SystemCheck likely has a 'prompt' or 'system_prompt' field
            # Let's check what fields are available
            if i == 1:
                print(f"Available fields: {list(item.keys())}")
            
            # SystemCheck uses 'instructions' field for the system prompt
            if 'instructions' in item and item['instructions']:
                prompt_text = item['instructions'].strip()
            elif 'description' in item and item['description']:
                # Fallback to description if instructions is empty
                prompt_text = item['description'].strip()
            else:
                # Skip items without prompt text (only print warning for first few)
                if i <= 10:
                    print(f"Warning: No prompt text found for item {i}")
                continue
            
            # Format like system_prompts_generated.txt
            output_lines.append(f"Prompt {i}:")
            output_lines.append(prompt_text)
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # Write to file
        output_file = "setup_data/systemcheck_prompts.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\nConverted {len(data)} prompts to {output_file}")
        print(f"Format matches system_prompts_generated.txt")
        
        # Show a sample
        print(f"\nSample of first prompt:")
        print("=" * 50)
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:10]:  # Show first 10 lines
                print(line.rstrip())
        
        return output_file
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    download_systemcheck()