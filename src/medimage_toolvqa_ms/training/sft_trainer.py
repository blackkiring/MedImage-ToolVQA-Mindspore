from __future__ import annotations

from typing import Any


class MindSporeSFTTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def dry_run_summary(self) -> dict[str, Any]:
        training = self.config.get("training", {})
        mindspore = self.config.get("mindspore", {})
        return {
            "model_name_or_path": training.get("model_name_or_path"),
            "train_records": training.get("train_records"),
            "output_dir": training.get("output_dir"),
            "device_target": mindspore.get("device_target"),
            "status": "mindspore_training_pending",
        }

    def train(self) -> None:
        raise NotImplementedError(
            "MindSpore SFT training implementation is pending. "
            "Use --dry-run to validate configuration loading."
        )
