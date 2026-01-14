# Citation SFT

Fine-tune Qwen3-1.7B to generate answers with proper document citations using the ALCE benchmark.

## Overview

This project trains an LLM to answer questions using provided documents and cite them correctly using `[1]`, `[2]`, etc. notation.

## Dataset

The dataset is built from [ALCE](https://github.com/princeton-nlp/ALCE) human evaluation data, which contains high-quality question-answer pairs with verified citations.

**Building the dataset:**
```bash
python -m citation_sft.sft_data.build_dataset
```

This will:
1. Download ALCE data and human eval annotations
2. Join examples by question ID
3. Filter for perfect precision/recall scores (55 examples)
4. Apply data augmentation (see below)
5. Save to `data/sft_dataset.json`

The dataset is pre-built and committed to this repo - no need to regenerate unless modifying augmentation.

### Data Augmentation

To teach the model to ignore irrelevant documents, I augment each example by inserting 1-2 random documents from other questions at random positions. Citation indices in labels are remapped accordingly.

Example: Original has docs `[1]-[5]` → Insert irrelevant doc at position 3 → Original `[3]` becomes `[4]`, `[4]` becomes `[5]`, etc.

Final dataset: 55 base + 110 augmented = 165 examples.

## Setup
```bash
uv sync
```

## Training

**Local:**
```bash
python -m citation_sft.train
```

**SLURM cluster (2 nodes, 4 GPUs):**
```bash
sbatch train.slurm
```

Logs are saved to `data/runs/` (TensorBoard format).

## Project Structure
```
citation_sft/
├── train.py                 # Training script
├── settings.py              # Pydantic settings
├── prompts/                 # Prompt templates
│   ├── citation_sft.txt
│   └── document.txt
└── sft_data/
    ├── build_dataset.py     # Dataset generation
    ├── sft_data.py          # SFTExample, InputDoc
    ├── _alce_models.py      # ALCE data models
    └── _human_eval_models.py
```

## References

- [ALCE Paper](https://arxiv.org/abs/2305.14627): Enabling Large Language Models to Generate Text with Citations
- [ALCE Repo](https://github.com/princeton-nlp/ALCE)