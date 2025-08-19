# KL Divergence Phrase Importance Analysis - Implementation Guide

## Overview
This implementation provides a comprehensive framework for identifying important phrases in LLM system prompts using KL divergence with controlled paraphrastic perturbations.

## Key Improvements Implemented

### 1. **Statistical Robustness**
- **Bootstrap Confidence Intervals**: 95% CI for all importance scores
- **Effect Size Calculation**: Cohen's d to quantify practical significance
- **Multiple Testing Correction**: FDR (Benjamini-Hochberg) for identifying truly significant phrases
- **Directional Analysis**: Tracks whether paraphrases increase/decrease model confidence

### 2. **Enhanced Scoring Mechanism**
- **Normalized Importance Scores**: Accounts for phrase length, effect size, confidence, and semantic similarity
- **Jensen-Shannon Divergence Option**: More stable than raw KL divergence
- **Vocabulary-normalized KL**: Scales by effective vocabulary size

### 3. **Advanced Paraphrase Generation**
- **Dual Methods**: Embedding-based (fast, free) and LLM-based (creative, high-quality)
- **Semantic Similarity Control**: Ensures paraphrases stay within controlled bounds (0.3-0.9 similarity)
- **Multi-strategy Generation**: Lexical, syntactic, and pragmatic variants
- **Quality Filtering**: Duplicate detection, similarity validation, meaning preservation

### 4. **Computational Efficiency**
- **Parallel Processing**: Analyzes multiple phrases concurrently
- **100-token Truncation**: Focuses on most informative output portions
- **Monte Carlo Sampling**: Reduces variance with multiple samples

### 5. **Phrase Interaction Analysis**
- **Pairwise Interaction Matrix**: Detects synergistic/antagonistic effects
- **Deviation from Additivity**: Measures non-linear phrase combinations

## File Structure

```
phrases_importance/
├── setup_data/
│   ├── extract_data.py              # Phrase extraction with regex patterns
│   ├── find_similar_phrases.py      # Embedding-based paraphrase generation
│   ├── llm_paraphrase_generator.py  # LLM-based paraphrase generation
│   └── [data files]
├── prompts/
│   ├── user_prompts.py          # Standardized test prompts
│   └── [prompt files]
├── kl_divergence/
│   ├── kl_computer.py           # Core KL computation with improvements
│   └── api_setup.py
├── evaluation/
│   ├── kl_generator.py
│   └── ranking.py
└── comprehensive_kl_analysis.py # Main analysis script with all enhancements

```

## Usage

### Basic Analysis (Embedding-based)
```bash
# Set API key
export TOGETHER_API_KEY='your-key-here'

# Run with embedding-based paraphrases (default)
python comprehensive_kl_analysis.py
```

### Advanced Analysis (LLM-based)
```bash
# 1. Generate LLM paraphrases first
python setup_data/llm_paraphrase_generator.py

# 2. Run analysis with LLM paraphrases
python comprehensive_kl_analysis.py \
    --paraphrase-method llm \
    --llm-paraphrase-file llm_paraphrases_together_20240101.json
```

### Custom Analysis
```python
from comprehensive_kl_analysis import EnhancedPhraseImportanceAnalyzer

# Initialize analyzer
analyzer = EnhancedPhraseImportanceAnalyzer(
    model_name="deepseek-ai/deepseek-v2.5",
    num_monte_carlo=3,
    bootstrap_iterations=1000,
    paraphrase_method="llm"  # or "embedding"
)

# Analyze phrases
results = analyzer.analyze_phrases_parallel(
    system_prompt, phrases, user_prompts
)

# Get normalized rankings
rankings = analyzer.compute_normalized_importance_scores(results)

# Statistical testing
significant = analyzer.perform_statistical_testing(results)
```

## Key Algorithms

### 1. KL Divergence with Jensen-Shannon Smoothing
```python
JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5(P+Q)
```

### 2. Normalized Importance Score
```
Score = base_kl * length_factor * effect_weight * confidence_weight * similarity_penalty
```

### 3. Bootstrap Confidence Interval
- Resample KL scores with replacement
- Compute percentile intervals (2.5%, 97.5%)

## Output

The analysis generates:
1. **detailed_results.json**: Complete numerical results
2. **phrase_importance.png**: Bar chart of top phrases
3. **effect_sizes.png**: Scatter plot of effect sizes
4. **phrase_interactions.png**: Heatmap of interactions
5. **analysis_report.txt**: Comprehensive text report

## Advantages Over Existing Methods

1. **No OOD Perturbations**: Uses semantically similar phrases from the same domain
2. **Statistical Guarantees**: Confidence intervals and significance testing
3. **Interaction Detection**: Captures non-additive phrase effects
4. **Interpretable Scores**: Normalized and effect-size weighted
5. **Computational Efficiency**: Parallel processing and smart truncation

## Future Enhancements

1. **Adaptive Sampling**: Dynamically adjust Monte Carlo samples based on variance
2. **Hierarchical Phrase Analysis**: Group phrases by category
3. **Cross-model Validation**: Compare importance across different LLMs
4. **Temporal Analysis**: Track phrase importance over conversation turns
5. **Causal Analysis**: Use do-calculus for true causal effects