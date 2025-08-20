#!/usr/bin/env python3
"""
Download first 500 entries from SimpleMix config of SystemCheck dataset.
"""

from datasets import load_dataset

def download_simplemix_500():
    """Download first 500 SimpleMix prompts from SystemCheck dataset."""
    
    print("Loading SystemCheck SimpleMix dataset from HuggingFace...")
    
    try:
        # Load the SimpleMix configuration
        dataset = load_dataset("normster/SystemCheck", "simplemix")
        
        # Get the data
        data = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        
        print(f"Total entries in SimpleMix: {len(data)}")
        print(f"Downloading first 500 entries...")
        
        # Convert to the same format as system_prompts_generated.txt
        output_lines = []
        valid_prompts = 0
        
        # Process only first 500 entries
        for i, item in enumerate(data):
            if i >= 500:  # Stop after 500 entries
                break
                
            # SimpleMix uses 'messages' format with system prompts
            system_prompt = None
            
            if 'messages' in item and item['messages']:
                # Look for system message
                for msg in item['messages']:
                    if msg.get('role') == 'system':
                        system_prompt = msg.get('content', '').strip()
                        break
            
            if not system_prompt:
                continue
            
            valid_prompts += 1
            
            # Format like system_prompts_generated.txt
            output_lines.append(f"Prompt {valid_prompts}:")
            output_lines.append(system_prompt)
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # Write to file
        output_file = "setup_data/simplemix_prompts_500.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\nSuccessfully converted {valid_prompts} prompts to {output_file}")
        
        # Show statistics
        print(f"\n=== STATISTICS ===")
        print(f"Entries processed: 500")
        print(f"Valid system prompts found: {valid_prompts}")
        
        # Show first 3 prompts as sample
        print(f"\n=== SAMPLE PROMPTS ===")
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            prompts = content.split("=" * 80)[:4]  # Get first 3 prompts (4th is empty)
            
            for i, prompt in enumerate(prompts[:3]):
                if prompt.strip():
                    lines = prompt.strip().split('\n')
                    print(f"\n{lines[0]}")
                    print(f"Length: {len(lines[1])} chars")
                    print(f"Preview: {lines[1][:150]}...")
        
        return output_file
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    download_simplemix_500()