# KL Divergence Phrase Importance - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  comprehensive_kl_analysis.py                │
│                     (Main Analysis Pipeline)                 │
│  - Orchestrates entire analysis                             │
│  - Parallel processing                                      │
│  - Statistical testing & normalization                      │
│  - Visualization & reporting                                │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
        ┌─────────▼──────────┐  ┌────────▼──────────┐
        │  kl_computer.py    │  │ find_similar_phrases.py
        │  (Core Engine)     │  │ (Paraphrase Gen)   │
        │                    │  │                    │
        │ - API calls        │  │ - Embeddings      │
        │ - Token dist       │  │ - Similarity      │
        │ - KL computation   │  │ - Variants        │
        │ - Bootstrap CI     │  └────────┬──────────┘
        │ - Effect sizes     │           │
        └─────────┬──────────┘           │
                  │                       │
        ┌─────────▼───────────────────────▼──────────┐
        │              Data Layer                    │
        ├────────────────────────────────────────────┤
        │ • extracted_prompts.txt - System prompts   │
        │ • candidate_phrases_for_kl.txt - Phrases   │
        │ • embedding_based_paraphrases.txt         │
        │ • user_prompts.json - Test prompts        │
        └────────────────────────────────────────────┘
```

## Module Responsibilities

### 1. **comprehensive_kl_analysis.py** (Orchestrator)
Main entry point that coordinates the entire analysis:
- `EnhancedPhraseImportanceAnalyzer`: Main analysis class
- `analyze_phrases_parallel()`: Parallel phrase processing
- `compute_normalized_importance_scores()`: Multi-factor scoring
- `perform_statistical_testing()`: FDR-corrected significance
- `analyze_phrase_interactions()`: Pairwise effects
- `generate_comprehensive_report()`: Visualization & reporting

### 2. **kl_computer.py** (Computation Engine)
Low-level KL divergence computation:
- `KLDivergenceComputer100Token`: Core KL computer
- `get_token_distribution_approximation()`: Token sampling
- `compute_kl_divergence_approximation()`: KL/JS divergence
- `_compute_bootstrap_ci()`: Confidence intervals
- `_compute_effect_size()`: Cohen's d calculation
- API integration with Together/OpenAI/Anthropic

### 3. **find_similar_phrases.py** (Paraphrase Generation)
Embedding-based paraphrase generation:
- `EmbeddingBasedParaphraser`: Main paraphraser
- `generate_all_paraphrase_variants()`: Multi-strategy generation
- `compute_embeddings()`: Sentence embeddings
- Semantic similarity filtering (0.3-0.9 range)
- Diversity enforcement

### 4. **extract_data.py** (Phrase Extraction)
Initial phrase extraction from prompts:
- `PhraseExtractor`: Regex-based extraction
- Pattern matching for different phrase types
- Deduplication and validation

### 5. **user_prompts.py** (Test Data)
Standardized user prompts for testing:
- 20 prompts across 4 categories
- Consistent test scenarios

## Data Flow

1. **Setup Phase**:
   - Extract phrases from system prompts → `candidate_phrases_for_kl.txt`
   - Generate paraphrases → `embedding_based_paraphrases.txt`
   - Create user prompts → `user_prompts.json`

2. **Analysis Phase**:
   - Load all data into `comprehensive_kl_analysis.py`
   - For each phrase:
     - Get paraphrases with similarities
     - Compute KL divergence via `kl_computer.py`
     - Track statistics and confidence
   - Parallel processing for efficiency

3. **Output Phase**:
   - Statistical analysis and ranking
   - Generate visualizations
   - Create comprehensive report

## Key Improvements Over Standard Approaches

1. **Statistical Robustness**:
   - Bootstrap confidence intervals
   - Effect size quantification
   - Multiple testing correction

2. **Semantic Control**:
   - Paraphrases stay within semantic bounds
   - No OOD perturbations
   - Similarity tracking

3. **Comprehensive Scoring**:
   - Length normalization
   - Effect size weighting
   - Confidence weighting
   - Semantic similarity penalty

4. **Interaction Detection**:
   - Pairwise phrase effects
   - Non-linear combinations
   - Synergy/antagonism detection

## Usage

```bash
# Set API key
export TOGETHER_API_KEY='your-key-here'

# Run comprehensive analysis
python comprehensive_kl_analysis.py

# Results in analysis_results/:
# - detailed_results.json
# - phrase_importance.png
# - effect_sizes.png
# - phrase_interactions.png
# - analysis_report.txt
```