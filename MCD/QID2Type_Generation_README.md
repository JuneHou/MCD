# VQA-CP v2 Question ID to Question Type Mapping

## Overview

This document explains how to generate the `qid2type_cpv2.json` file required by the MCD (Margin-based Counterfactual Debiasing) model for VQA-CP v2 dataset evaluation.

## What is qid2type?

The `qid2type` mapping is a JSON file that maps question IDs to their corresponding question types. It's used during model evaluation to:

- Categorize questions by their types (e.g., "what", "how many", "is the")
- Enable detailed performance analysis across different question categories
- Calculate type-specific accuracy scores (yes/no, number, other)

## File Structure

### Input Files (VQA-CP v2 Annotations)
```
/data/wang/junh/datasets/vqa-cp-v2/
├── vqacp_v2_train_annotations.json  # Training annotations (438,183 questions)
└── vqacp_v2_test_annotations.json   # Test annotations (219,928 questions)
```

### Output File
```
/data/wang/junh/githubs/MCD_new/MCD/util/qid2type_cpv2.json
```

## Annotation File Format

Each annotation in the VQA-CP v2 files contains:

```json
{
  "question_id": 27511005,
  "question_type": "what does the",
  "answer_type": "other",
  "image_id": 27511,
  "coco_split": "train2014",
  "multiple_choice_answer": "lancashire united",
  "answers": [...]
}
```

Key fields for qid2type generation:
- `question_id`: Unique identifier (converted to string in mapping)
- `question_type`: Question category used for evaluation grouping

## Generated Mapping Format

The output `qid2type_cpv2.json` file format:

```json
{
  "27511005": "what does the",
  "334362002": "is the",
  "344029012": "what",
  "214729003": "what kind of",
  "366009022": "what are",
  ...
}
```

## Usage in MCD Model

The mapping is loaded in `main_MCD.py` during evaluation:

```python
if config.cp_data:
    if config.version=='v2':
        with open('/data/wang/junh/githubs/MCD/MCD/util/qid2type_cpv2.json', 'r') as f:
            qid2type = json.load(f)
    else:
        with open('../util/qid2type_cpv1.json', 'r') as f:
            qid2type = json.load(f)
else:
    with open('../util/qid2type_v2.json', 'r') as f:
        qid2type = json.load(f)
```

## Quick Generation Method

### Method 1: Using the Documented Script
```bash
python generate_qid2type_mapping.py
```

### Method 2: Quick One-liner
```python
import json

# Load annotations
train_ann = json.load(open('/data/wang/junh/datasets/vqa-cp-v2/vqacp_v2_train_annotations.json'))
test_ann = json.load(open('/data/wang/junh/datasets/vqa-cp-v2/vqacp_v2_test_annotations.json'))

# Create mapping
qid2type = {}
for ann in train_ann + test_ann:
    qid2type[str(ann['question_id'])] = ann['question_type']

# Save mapping
with open('/data/wang/junh/githubs/MCD_new/MCD/util/qid2type_cpv2.json', 'w') as f:
    json.dump(qid2type, f, indent=2)

print(f"Generated {len(qid2type)} mappings")
```

## Question Type Statistics

The VQA-CP v2 dataset contains 65 unique question types with the following distribution:

| Question Type | Count | Percentage |
|---------------|-------|------------|
| how many | 62,801 | 9.5% |
| is the | 52,192 | 7.9% |
| what | 50,505 | 7.7% |
| what color is the | 42,023 | 6.4% |
| what is the | 35,855 | 5.4% |
| none of the above | 25,523 | 3.9% |
| is this | 24,285 | 3.7% |
| is this a | 23,516 | 3.6% |
| what is | 19,889 | 3.0% |
| what kind of | 17,032 | 2.6% |
| ... | ... | ... |

**Total: 658,111 questions across train and test sets**

## Verification

The generated mapping should contain:
- ✅ All 438,183 training question IDs
- ✅ All 219,928 test question IDs  
- ✅ Total of 658,111 unique question IDs
- ✅ No missing or duplicate entries

## Troubleshooting

### Common Issues:

1. **File not found**: Ensure VQA-CP v2 annotation files are in the correct location
2. **Permission errors**: Check write permissions for the output directory
3. **JSON parsing errors**: Verify annotation files are valid JSON
4. **Missing mappings**: Run verification to check completeness

### File Locations for Different MCD Versions:

- **MCD_new**: `/data/wang/junh/githubs/MCD_new/MCD/util/qid2type_cpv2.json`
- **MCD**: `/data/wang/junh/githubs/MCD/MCD/util/qid2type_cpv2.json`

Make sure to generate the file in the correct location based on which MCD version you're using.
