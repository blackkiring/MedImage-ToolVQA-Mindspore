# MedImage-ToolVQA-Mindspore

MindSpore implementation scaffold for MedImage-ToolVQA. The repository is
organized around two primary tasks:

1. Dataset construction for medical image tool-use VQA records.
2. SFT fine-tuning with MindSpore.

The current code is a minimal, runnable scaffold. It defines configuration,
schemas, pipeline stages, and command-line entry points, while leaving full
MindSpore model training implementation for later work.

## Repository Layout

```text
configs/
  dataset_building.yaml           Dataset construction pipeline config
  sft_train.yaml                  MindSpore SFT config
  eval.yaml                       Auxiliary evaluation config
scripts/
  build_dataset.py                Dataset construction entry point
  train_sft.py                    SFT fine-tuning entry point
  evaluate.py                     Auxiliary evaluation entry point
src/medimage_toolvqa_ms/
  dataset_building/               merge/make_vqa/verify/reasoning/SFT stages
  sft/                            MindSpore SFT dataset adapter and trainer
  data/                           Shared record validation utilities
  tools/                          Tool schema definitions
  evaluation/                     Post-SFT evaluation scaffold
tests/
  test_config.py
  test_dataset_building.py
  test_schema.py
  test_sft.py
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate the dataset construction and SFT scaffolds:

```bash
python scripts/build_dataset.py --config configs/dataset_building.yaml --dry-run
python scripts/train_sft.py --config configs/sft_train.yaml --dry-run
python scripts/evaluate.py --config configs/eval.yaml --dry-run
pytest
```

If `pytest` is not installed, the first three dry-run commands still validate
configuration loading and core entry points.

## Dataset Construction

The dataset construction scaffold follows the MedImage-ToolVQA chapter flow:

```text
merge -> make_vqa -> verify -> makereasoning -> make_sft
```

Each stage is represented as a typed pipeline step. The initial implementation
does not call external VLMs or image tools; it records the expected input and
output locations so the concrete implementation can be added without changing
the repository layout.

## SFT Fine-Tuning

The SFT scaffold defines:

- training configuration loading,
- SFT record validation,
- MindSpore runtime settings,
- a trainer dry-run summary,
- an explicit pending marker for full MindSpore training.

The training entry point intentionally raises a clear error outside `--dry-run`
until the concrete MindSpore model, tokenizer, image processor, and optimizer
logic are implemented.

## Safety Scope

MedImage-ToolVQA is intended for research, data engineering, and model training
supervision. Outputs from models trained with this project must not be treated
as clinical diagnosis.
