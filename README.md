# MCD: Margin-based Counterfactual Debiasing for VQA

This repository contains the implementation of Margin-based Counterfactual Debiasing (MCD) for Visual Question Answering on the VQA-CP v2 dataset.

## Features

- **Improved GPU handling**: Use `--gpu` argument to specify which GPU to use
- **Complete qid2type mapping**: Pre-generated question ID to question type mapping for VQA-CP v2
- **Comprehensive documentation**: Detailed setup and usage instructions
- **Better error handling**: Improved path management and error messages

## Quick Start

### Prerequisites
- Python 3.6+
- PyTorch
- VQA-CP v2 dataset
- Pre-trained GloVe embeddings

### Basic Training
```bash
# Train with default settings (GPU 0)
python MCD/main_MCD.py

# Train on specific GPU
python MCD/main_MCD.py --gpu 2

# Train with custom parameters
python MCD/main_MCD.py --epochs 50 --lr 0.001 --batch-size 256 --gpu 1
```

### Evaluation Only
```bash
# Evaluate on validation set
python MCD/main_MCD.py --eval-only --resume --name logs/model.pth --gpu 1

# Test on test set
python MCD/main_MCD.py --test --resume --name logs/model.pth --gpu 1
```

### Fine-tuning
```bash
python MCD/main_MCD.py --fine-tune --resume --name logs/pretrained.pth --name-new finetuned.pth --gpu 1
```

## Model Options

- **`baseline`**: Standard attention mechanism with margin learning
- **`baseline_newatt`**: Enhanced new attention mechanism with margin learning (default, recommended)

Both models use the ArcMarginProduct for margin-based learning. The main difference is in the attention mechanism:
- `baseline`: Uses concatenation-based attention  
- `baseline_newatt`: Uses projection-based attention with element-wise multiplication

## Configuration

Key settings in `MCD/utils/config.py`:
- `cp_data = True`: Use VQA-CP dataset
- `version = 'v2'`: VQA-CP v2 
- `loss_type = 'ce_margin'`: Use margin-based loss
- `scale = 16`: Margin scaling factor

## Files Added/Modified

### New Documentation
- `MCD/QID2Type_Generation_README.md`: Guide for generating qid2type mappings
- `MCD/MCD_Replication_Investigation_Report.md`: Investigation report
- `MCD/generate_qid2type_mapping.py`: Script to generate qid2type mappings

### Key Improvements
- **GPU Selection**: Fixed `main_MCD.py` to properly use `--gpu` argument
- **qid2type Mapping**: Complete mapping for VQA-CP v2 (658,111 questions)
- **Better .gitignore**: Excludes model files, cache, and large data files

## Question Type Analysis

The VQA-CP v2 dataset contains 65 unique question types:
- **how many**: 62,801 questions (9.5%)
- **is the**: 52,192 questions (7.9%) 
- **what**: 50,505 questions (7.7%)
- **what color is the**: 42,023 questions (6.4%)
- And many more...

## Directory Structure

```
MCD/
├── main_MCD.py              # Main training/evaluation script
├── train_MCD.py             # Training loop implementation  
├── modules/                 # Model architectures
├── utils/                   # Utilities and configuration
├── util/                    # Generated mappings
│   └── qid2type_cpv2.json  # Question ID to type mapping
├── tools/                   # Data processing tools
└── docs/                    # Documentation
```

## Citation

If you use this code, please cite the original MCD paper and this repository.

## License

This project follows the same license as the original MCD implementation.
