from __future__ import annotations

from typing import Any

from medimage_toolvqa_ms.sft.dataset import sft_dataset_config_from_yaml


class MindSporeSFTTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_config = sft_dataset_config_from_yaml(config)

    def dry_run_summary(self) -> dict[str, Any]:
        training = self.config.get("training", {})
        mindspore = self.config.get("mindspore", {})
        return {
            "model_name_or_path": training.get("model_name_or_path"),
            "train_records": self.dataset_config.train_records,
            "eval_records": self.dataset_config.eval_records,
            "image_root": self.dataset_config.image_root,
            "output_dir": training.get("output_dir"),
            "device_target": mindspore.get("device_target"),
            "status": "mindspore_sft_training_pending",
        }

    def train(self) -> None:
        raise NotImplementedError(
            "MindSpore SFT training implementation is pending. "
            "Use --dry-run to validate SFT configuration loading."
        )
