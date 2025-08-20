#!/usr/bin/env python3
"""
Download unique system prompts from SimpleMix until we have ~300 unique prompts.
"""

from datasets import load_dataset

def download_unique_simplemix(target_count=300):
    """Download SimpleMix prompts, keeping only unique ones."""
    
    print(f"Loading SystemCheck SimpleMix dataset to get {target_count} unique prompts...")
    
    try:
        # Load the SimpleMix configuration
        dataset = load_dataset("normster/SystemCheck", "simplemix", streaming=True)
        
        # Get the data stream
        data_stream = dataset['train']
        
        # Track unique prompts
        unique_prompts = {}  # prompt_text -> first_occurrence_index
        processed = 0
        
        print("Processing entries to find unique system prompts...")
        
        # Process entries until we have enough unique prompts
        for item in data_stream:
            processed += 1
            
            # Extract system prompt
            system_prompt = None
            if 'messages' in item and item['messages']:
                for msg in item['messages']:
                    if msg.get('role') == 'system':
                        system_prompt = msg.get('content', '').strip()
                        break
            
            if system_prompt:
                # Check if this is a new unique prompt
                if system_prompt not in unique_prompts:
                    unique_prompts[system_prompt] = len(unique_prompts) + 1
                    
                    if len(unique_prompts) % 50 == 0:
                        print(f"  Found {len(unique_prompts)} unique prompts (processed {processed} entries)")
                    
                    # Stop when we reach target
                    if len(unique_prompts) >= target_count:
                        break
            
            # Safety limit to avoid processing entire dataset
            if processed >= 50000:
                print(f"Processed {processed} entries, stopping early")
                break
        
        print(f"\nProcessed {processed} entries")
        print(f"Found {len(unique_prompts)} unique system prompts")
        
        # Convert to output format
        output_lines = []
        for prompt_text, prompt_num in sorted(unique_prompts.items(), key=lambda x: x[1]):
            output_lines.append(f"Prompt {prompt_num}:")
            output_lines.append(prompt_text)
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # Write to file
        output_file = "setup_data/simplemix_unique_prompts.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\nSaved {len(unique_prompts)} unique prompts to {output_file}")
        
        # Analyze prompt characteristics
        prompt_lengths = [len(p) for p in unique_prompts.keys()]
        
        print(f"\n=== PROMPT STATISTICS ===")
        print(f"Total unique prompts: {len(unique_prompts)}")
        print(f"Length range: {min(prompt_lengths)} - {max(prompt_lengths)} chars")
        print(f"Average length: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
        print(f"Median length: {sorted(prompt_lengths)[len(prompt_lengths)//2]} chars")
        
        # Show length distribution
        print("\nLength distribution:")
        ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]
        for low, high in ranges:
            count = sum(1 for l in prompt_lengths if low <= l < high)
            if count > 0:
                label = f"{low}-{high}" if high != float('inf') else f"{low}+"
                print(f"  {label} chars: {count} prompts ({count/len(unique_prompts)*100:.1f}%)")
        
        # Show sample prompts
        print(f"\n=== SAMPLE PROMPTS ===")
        sample_prompts = list(unique_prompts.keys())[:3]
        for i, prompt in enumerate(sample_prompts, 1):
            print(f"\nPrompt {i}:")
            print(f"Length: {len(prompt)} chars")
            if len(prompt) > 200:
                print(f"Text: {prompt[:200]}...")
            else:
                print(f"Text: {prompt}")
        
        return output_file
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    download_unique_simplemix(target_count=300)