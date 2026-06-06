# MedImage-ToolVQA-Mindspore

MindSpore implementation scaffold for MedImage-ToolVQA, a medical image
tool-use VQA data engineering project.

This repository currently provides the minimum project structure for future
MindSpore data processing, SFT training, and evaluation work. It does not yet
ship a complete model training implementation.

## Repository Layout

```text
configs/                         Example YAML configuration files
scripts/                         Command-line entry points
src/medimage_toolvqa_ms/          Python package
tests/                           Minimal regression tests
```

## Quick Start

Install basic dependencies:

```bash
pip install -r requirements.txt
```

Validate the scaffold:

```bash
python scripts/prepare_data.py --config configs/dataset.yaml --dry-run
python scripts/train_sft.py --config configs/sft_train.yaml --dry-run
python scripts/evaluate.py --config configs/eval.yaml --dry-run
pytest
```

## Status

- Data schema and tool schema: scaffolded.
- Data preparation entry point: scaffolded.
- MindSpore SFT training entry point: scaffolded.
- Evaluation entry point: scaffolded.
- Full MindSpore model training logic: pending.

## Safety Scope

MedImage-ToolVQA is intended for research, data engineering, and model training
supervision. Outputs from models trained with this project must not be treated
as clinical diagnosis.
