# Citation SFT

Fine-tune Qwen3-1.7B to generate answers with proper document citations using the ALCE benchmark.

## Overview

This project trains an LLM to answer questions using provided documents and cite them correctly using `[1]`, `[2]`, etc. notation.

## Dataset

The dataset is built from [ALCE](https://github.com/princeton-nlp/ALCE) human evaluation data, which contains high-quality question-answer pairs with verified citations.

**Building the dataset:**
```bash
uv run python src/citation_sft/sft_data/build_dataset.py
```

This will:
1. Download ALCE data and human eval annotations
2. Join examples by question ID
3. Filter for perfect precision/recall scores (55 examples)
4. Apply data augmentation (see below)
5. Save to `data/sft_dataset.json`

The dataset is pre-built and committed to this repo - no need to regenerate unless modifying augmentation.

### Data Augmentation

To teach the model to ignore irrelevant documents, we augment each example by inserting 1-2 random documents from other questions at random positions. Citation indices in labels are remapped accordingly.

Example: Original has docs `[1]-[5]` → Insert irrelevant doc at position 3 → Original `[3]` becomes `[4]`, `[4]` becomes `[5]`, etc.

Final dataset: 55 base + 110 augmented = 165 examples.

## Setup
```bash
uv sync
```

## Configuration

Training configs can be set via environment variables or `.env` file. See `.env.example` for all options.

Key parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `SFT_MODEL_NAME` | `Qwen/Qwen3-1.7B` | Model to fine-tune |
| `SFT_NUM_TRAIN_EPOCHS` | 3 | Training epochs |
| `SFT_LEARNING_RATE` | 2e-5 | Learning rate |
| `SFT_MAX_AUGMENT` | 2 | Max irrelevant docs to insert |

## Training

**Local:**
```bash
uv run python -m citation_sft.train
```

**SLURM cluster (2 nodes, 4 GPUs):**
```bash
# Update the path in train.slurm first
sbatch train.slurm
```

> Note: Update `cd /path/to/citation-sft` in `train.slurm` to your actual repo path before running.

Logs are saved to `data/runs/` (TensorBoard format).
```bash
tensorboard --logdir=data/runs
```

## Project Structure
```
citation-sft/
├── src/
│   └── citation_sft/
│       ├── train.py                 # Training script
│       ├── settings.py              # Pydantic settings
│       ├── prompts/                 # Prompt templates
│       │   ├── citation_sft.txt
│       │   └── document.txt
│       └── sft_data/
│           ├── build_dataset.py     # Dataset generation
│           ├── sft_data.py          # SFTExample, InputDoc
│           ├── _alce_models.py      # ALCE data models
│           └── _human_eval_models.py
├── data/
│   └── sft_dataset.json             # Pre-built dataset (committed)
├── train.slurm                      # SLURM submission script
├── .env.train                       # Train configuration
└── pyproject.toml
```

## References

- [ALCE Paper](https://arxiv.org/abs/2305.14627): Enabling Large Language Models to Generate Text with Citations
- [ALCE Repo](https://github.com/princeton-nlp/ALCE)