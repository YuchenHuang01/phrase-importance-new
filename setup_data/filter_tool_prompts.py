#!/usr/bin/env python3
"""
Filter out prompts that mention "tools" followed by JSON-like content.
"""

import re

def filter_tool_prompts(input_file="setup_data/systemmix_moderate_prompts.txt", 
                       output_file="setup_data/systemmix_moderate_no_tools.txt"):
    """Remove prompts that mention tools with JSON content."""
    
    print(f"Reading prompts from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator
    prompt_blocks = content.split("=" * 80)
    
    kept_prompts = []
    removed_prompts = []
    
    for block in prompt_blocks:
        block = block.strip()
        if not block or not block.startswith("Prompt"):
            continue
        
        # Check if this prompt mentions "tool" and has JSON-like content
        block_lower = block.lower()
        
        # Pattern to detect "tool" followed by JSON structures
        has_tool_json = False
        
        # Check for various patterns that indicate tool + JSON
        if "tool" in block_lower:
            # Check for JSON indicators after "tool" mention
            if any(pattern in block for pattern in [
                '{"', '{ "',  # JSON object start
                '[\n', '[ \n',  # JSON array start
                '"type":', '"name":', '"description":',  # Common JSON fields
                'json', 'JSON'  # Explicit JSON mention
            ]):
                has_tool_json = True
            
            # Also check for specific patterns like "tools: [" or "tool: {"
            if re.search(r'tools?\s*:\s*[\[{]', block, re.IGNORECASE):
                has_tool_json = True
        
        if has_tool_json:
            removed_prompts.append(block)
        else:
            kept_prompts.append(block)
    
    print(f"\nFiltering results:")
    print(f"  Original prompts: {len(kept_prompts) + len(removed_prompts)}")
    print(f"  Removed (with tool+JSON): {len(removed_prompts)}")
    print(f"  Kept: {len(kept_prompts)}")
    
    # Renumber the kept prompts
    output_lines = []
    for i, prompt_block in enumerate(kept_prompts, 1):
        lines = prompt_block.split('\n')
        # Replace the prompt number
        lines[0] = f"Prompt {i}:"
        output_lines.extend(lines)
        output_lines.append("=" * 80)
        output_lines.append("")
    
    # Write filtered prompts
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nSaved {len(kept_prompts)} filtered prompts to {output_file}")
    
    # Show examples of removed prompts
    if removed_prompts:
        print("\n=== SAMPLE REMOVED PROMPTS ===")
        for i, block in enumerate(removed_prompts[:3], 1):
            lines = block.split('\n')
            prompt_text = '\n'.join(lines[1:]).strip()  # Skip the "Prompt N:" line
            
            print(f"\nRemoved example {i}:")
            # Find and show the part with "tool"
            tool_index = prompt_text.lower().find('tool')
            if tool_index != -1:
                start = max(0, tool_index - 50)
                end = min(len(prompt_text), tool_index + 200)
                excerpt = prompt_text[start:end]
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(prompt_text):
                    excerpt = excerpt + "..."
                print(f"Excerpt: {excerpt}")

if __name__ == "__main__":
    filter_tool_prompts()