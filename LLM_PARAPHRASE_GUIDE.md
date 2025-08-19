# LLM Paraphrase Generation Guide

## Overview

The system now supports two methods for generating paraphrases:

1. **Embedding-based** (default): Uses sentence embeddings to find similar phrases from your dataset
2. **LLM-based**: Uses GPT-4, Claude, or other LLMs to generate novel paraphrases

## LLM Paraphrase Generation

### Step 1: Generate LLM Paraphrases

```bash
# Set your API key (choose one)
export TOGETHER_API_KEY='your-key'      # For Llama, DeepSeek via Together
export OPENAI_API_KEY='your-key'        # For GPT-4
export ANTHROPIC_API_KEY='your-key'     # For Claude

# Generate paraphrases
python setup_data/llm_paraphrase_generator.py
```

This will:
- Load phrases from `candidate_phrases_for_kl.txt`
- Prompt you to select an LLM provider
- Generate 5 paraphrases per phrase
- Filter by semantic similarity (0.5-0.9)
- Save results to `llm_paraphrases_[provider]_[timestamp].json`

### Step 2: Run KL Analysis with LLM Paraphrases

```bash
# Option 1: Use pre-generated LLM paraphrases
python comprehensive_kl_analysis.py \
    --paraphrase-method llm \
    --llm-paraphrase-file llm_paraphrases_together_20240101_120000.json

# Option 2: Generate LLM paraphrases on-demand (slower)
python comprehensive_kl_analysis.py --paraphrase-method llm
```

## Comparison: Embedding vs LLM Paraphrases

### Embedding-based Paraphrases
```python
# Example output:
Original: "helpful and harmless"
Paraphrases:
1. "helpful and honest" (sim: 0.82)
2. "accurate and reliable" (sim: 0.75)
3. "professional and thorough" (sim: 0.68)
```

**Pros:**
- ✅ Free (no API costs)
- ✅ Fast (milliseconds)
- ✅ Uses real phrases from your dataset
- ✅ Guaranteed semantic similarity

**Cons:**
- ❌ Limited to existing phrases
- ❌ Less creative variation

### LLM-generated Paraphrases
```python
# Example output:
Original: "helpful and harmless"
Paraphrases:
1. "beneficial while avoiding any potential harm" (sim: 0.87)
2. "supportive and safe in all interactions" (sim: 0.84)
3. "useful without causing negative effects" (sim: 0.81)
4. "constructive and non-detrimental" (sim: 0.76)
5. "assistive yet completely benign" (sim: 0.72)
```

**Pros:**
- ✅ Novel, creative variations
- ✅ Better coverage of semantic space
- ✅ Can handle any phrase
- ✅ Multiple linguistic strategies

**Cons:**
- ❌ Costs money (~$0.001-0.004 per phrase)
- ❌ Slower (1-2 seconds per phrase)
- ❌ Requires API keys

## LLM Paraphrase Quality Control

The LLM generator includes several quality controls:

1. **Semantic Similarity Filtering**
   - Only keeps paraphrases with similarity 0.5-0.9
   - Too similar (>0.9) = not enough variation
   - Too different (<0.5) = meaning drift

2. **Multiple Strategies**
   - Lexical: Word substitutions
   - Syntactic: Structure variations
   - Pragmatic: Tone/formality shifts

3. **Duplicate Detection**
   - Case-insensitive deduplication
   - Filters trivial variations

## Cost Estimation

| Provider | Model | Cost per Phrase | 100 Phrases |
|----------|-------|-----------------|-------------|
| Together | Llama-3.1-70B | ~$0.001 | ~$0.10 |
| OpenAI | GPT-4-Turbo | ~$0.003 | ~$0.30 |
| Anthropic | Claude-3-Opus | ~$0.004 | ~$0.40 |

## Best Practices

1. **For Initial Testing**: Use embedding-based (free, fast)
2. **For Production Analysis**: Use LLM-based for better coverage
3. **Hybrid Approach**: 
   - Use embeddings for common phrases
   - Use LLM for rare/complex phrases

## Example Workflow

```bash
# 1. Extract phrases
python setup_data/extract_data.py

# 2. Generate LLM paraphrases (optional)
python setup_data/llm_paraphrase_generator.py
# Select: 1 (Together/Llama)
# Phrases: 20
# Continue: y

# 3. Run analysis with LLM paraphrases
python comprehensive_kl_analysis.py \
    --paraphrase-method llm \
    --llm-paraphrase-file llm_paraphrases_together_20240101_120000.json

# 4. View results
# Results in analysis_results/
# - detailed_results.json
# - phrase_importance.png
# - analysis_report.txt
```

## Troubleshooting

1. **"No API key found"**
   ```bash
   export TOGETHER_API_KEY='your-key-here'
   ```

2. **"Rate limit exceeded"**
   - Add delays between API calls
   - Reduce batch size

3. **"Low quality paraphrases"**
   - Adjust temperature (0.5-0.9)
   - Try different model
   - Check similarity thresholds

## Advanced Usage

### Custom Paraphrase Strategies
```python
from setup_data.llm_paraphrase_generator import LLMParaphraseGenerator, ParaphraseConfig

config = ParaphraseConfig(
    provider="openai",
    model_name="gpt-4-turbo-preview",
    temperature=0.8,  # More creative
    min_similarity=0.6,  # Allow more variation
    max_similarity=0.85
)

generator = LLMParaphraseGenerator(config)

# Generate with specific strategy
paraphrases = generator.generate_paraphrases(
    "avoid harmful content",
    num_variants=7,
    strategy="syntactic"  # Focus on structure variations
)
```

### Batch Processing with Progress
```python
# Process large phrase sets
results = generator.batch_generate_paraphrases(
    phrases[:100],  # First 100 phrases
    num_variants=5
)

# Save incrementally
generator.save_results(results, "batch_1_paraphrases.json")
```