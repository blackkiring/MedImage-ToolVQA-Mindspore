from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFTDatasetConfig:
    train_records: str
    eval_records: str | None
    image_root: str


def sft_dataset_config_from_yaml(config: dict[str, Any]) -> SFTDatasetConfig:
    training = config.get("training", {})
    sft_data = config.get("sft_data", {})
    return SFTDatasetConfig(
        train_records=training.get("train_records", ""),
        eval_records=training.get("eval_records"),
        image_root=sft_data.get("image_root", "data/images"),
    )
