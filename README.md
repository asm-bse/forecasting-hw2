# Climate Claim Classification: Zero-Shot vs. Few-Shot Learning

This project demonstrates and compares two advanced Natural Language Processing (NLP) paradigms – **Zero-Shot Learning (ZSL)** and **Few-Shot Learning (FSL)** – for the task of classifying climate-related claims.

## Overview

The goal is to distinguish, with claim-evidence pairs, whether the claim is related (1) or unrelated (0) to the evidence using data from the [climate-claim-related dataset](https://huggingface.co/datasets/mwong/climate-claim-related).

We evaluate 7 different models:
- **3 Zero-shot learning models** with natural language inference (NLI)
- **4 Few-shot learning models** using contrastive learning (SetFit) and standard fine-tuning

## Setup

### Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU recommended** for fine-tuning models (especially the standard fine-tuning approaches)
- **16GB+ RAM** recommended for processing larger models
- Adjust batch sizes in the code if you encounter memory issues

## Running the Code

### Option 1: Run the Complete Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook hw2.ipynb
```

Execute all cells sequentially. **Warning**: The complete execution can take several hours, especially for the fine-tuning sections.

### Option 2: Run Specific Sections

The notebook is organized into clear sections that can be run independently:

1. **Data Loading and Preparation** - Always run this first
2. **Zero-Shot Learning (ZSL) with NLI** - Quick to run
3. **Few-Shot Learning (FSL) with SetFit** - Moderate time (~10-20 minutes)
4. **Standard Fine-tuning** - Longest time (~1-2 hours depending on hardware)

### Key Parameters to Adjust

If you encounter memory issues, modify these parameters in the code:

- `batch_size`: Reduce from 64 to 32 or 16
- `max_length`: Reduce sequence length for transformer models
- `per_device_train_batch_size` and `per_device_eval_batch_size`: Reduce for fine-tuning

## Outputs and Results

### Directory Structure

After running the code, the following directories will be created:

```
forecasting-hw2/
├── results/           # All evaluation results and visualizations
├── models/           # Saved trained models (folder created if notebook is executed)
├── hw2.ipynb        # Main notebook
├── README.md        # This file
└── requirements.txt # Dependencies
```

### Results Directory

The `results/` directory contains:

#### Evaluation Metrics and Visualizations
- **Confusion matrices**: `*_confusion_matrix.png`
- **ROC curves**: `*_roc_curve.png` 
- **Precision-Recall curves**: `*_pr_curve.png`

#### Model-Specific Results

**Zero-Shot Learning:**
- `zero_shot_nli_*` - Default NLI template results
- `zero_shot_nli_custom_template_*` - Custom template results  
- `zero_shot_nli_only_claims_*` - Claims-only results

**SetFit Models:**

- `setfit_16_*` - SetFit with 16 training examples
- `setfit_32_*` - SetFit with 32 training examples
- `setfit_results_*.parquet` - Training iteration results

**Fine-tuned Models:**

- `finetuned_albert-xlarge-vitaminc-mnli_frozen_16_*` - Frozen transformer layers
- `finetuned_albert-xlarge-vitaminc-mnli_unfrozen_16_*` - Full fine-tuning
- `cls_fine_tuning_results_*.parquet` - Training logs

### Models Directory

The `models/` directory contains:
- `setfit_best_*` - Best SetFit models for each configuration
- `cls_fine_tuning_*` - Fine-tuned transformer models and checkpoints

## Model Approaches

### Zero-Shot Learning (ZSL)
Uses pre-trained NLI models without any task-specific training:
- Default entailment template
- Custom hypothesis template  
- Claims-only classification

### Few-Shot Learning (FSL)

#### SetFit (Contrastive Learning)
- Uses sentence transformers with contrastive learning
- Tested with 16 and 32 labeled examples
- Multiple training iterations to reduce randomness

#### Standard Fine-tuning
- Uses ALBERT model pre-trained on fact-checking data
- Two approaches: frozen vs. unfrozen transformer layers
- Early stopping based on validation loss

## Performance Evaluation

Each model is evaluated using:
- **Classification metrics**: Accuracy, F1-score, Precision, Recall
- **Confusion matrices**: Visual representation of predictions
- **ROC curves**: True/False positive rate analysis
- **Precision-Recall curves**: Precision vs. recall trade-offs