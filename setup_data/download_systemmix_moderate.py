#!/usr/bin/env python3
"""
Download unique system prompts from SystemMix, filtering for moderate length.
"""

from datasets import load_dataset

def download_systemmix_moderate(target_count=300, min_length=100, max_length=1000):
    """Download SystemMix prompts, keeping moderate-length unique ones."""
    
    print(f"Loading SystemCheck SystemMix dataset...")
    print(f"Target: {target_count} unique prompts between {min_length}-{max_length} chars")
    
    try:
        # Load the SystemMix configuration
        dataset = load_dataset("normster/SystemCheck", "systemmix", streaming=True)
        
        # Get the data stream
        data_stream = dataset['train']
        
        # Track unique prompts
        unique_prompts = {}  # prompt_text -> first_occurrence_index
        processed = 0
        too_short = 0
        too_long = 0
        
        print("\nProcessing entries to find moderate-length unique system prompts...")
        
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
                prompt_length = len(system_prompt)
                
                # Filter by length
                if prompt_length < min_length:
                    too_short += 1
                    continue
                elif prompt_length > max_length:
                    too_long += 1
                    continue
                
                # Check if this is a new unique prompt
                if system_prompt not in unique_prompts:
                    unique_prompts[system_prompt] = len(unique_prompts) + 1
                    
                    if len(unique_prompts) % 50 == 0:
                        print(f"  Found {len(unique_prompts)} unique prompts (processed {processed} entries)")
                    
                    # Stop when we reach target
                    if len(unique_prompts) >= target_count:
                        break
            
            # Status update every 5000 entries
            if processed % 5000 == 0:
                print(f"  Processed {processed} entries... ({len(unique_prompts)} unique found)")
            
            # Safety limit
            if processed >= 100000:
                print(f"Processed {processed} entries, stopping early")
                break
        
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total entries processed: {processed}")
        print(f"Too short (<{min_length} chars): {too_short}")
        print(f"Too long (>{max_length} chars): {too_long}")
        print(f"Unique prompts found: {len(unique_prompts)}")
        
        # Convert to output format
        output_lines = []
        for prompt_text, prompt_num in sorted(unique_prompts.items(), key=lambda x: x[1]):
            output_lines.append(f"Prompt {prompt_num}:")
            output_lines.append(prompt_text)
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # Write to file
        output_file = "setup_data/systemmix_moderate_prompts.txt"
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
        ranges = [(100, 200), (200, 300), (300, 400), (400, 500), (500, 750), (750, 1000)]
        for low, high in ranges:
            count = sum(1 for l in prompt_lengths if low <= l < high)
            if count > 0:
                print(f"  {low}-{high} chars: {count} prompts ({count/len(unique_prompts)*100:.1f}%)")
        
        # Show sample prompts of different lengths
        print(f"\n=== SAMPLE PROMPTS ===")
        
        # Sort by length to show variety
        sorted_prompts = sorted(unique_prompts.items(), key=lambda x: len(x[0]))
        
        # Show shortest, median, and longest
        samples = [
            ("Shortest", sorted_prompts[0][0]),
            ("Median", sorted_prompts[len(sorted_prompts)//2][0]),
            ("Longest", sorted_prompts[-1][0])
        ]
        
        for label, prompt in samples:
            print(f"\n{label} prompt ({len(prompt)} chars):")
            if len(prompt) > 300:
                print(f"{prompt[:300]}...")
            else:
                print(prompt)
        
        return output_file
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Adjust these parameters as needed
    download_systemmix_moderate(
        target_count=300,      # How many unique prompts to collect
        min_length=100,        # Minimum prompt length (filter out very short)
        max_length=1000       # Maximum prompt length (filter out very long)
    )