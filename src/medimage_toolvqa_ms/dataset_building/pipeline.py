from __future__ import annotations

from typing import Any

from medimage_toolvqa_ms.dataset_building.stages import PipelineStage, build_default_stages


class DatasetBuildPipeline:
    def __init__(self, config: dict[str, Any], stages: tuple[PipelineStage, ...] | None = None) -> None:
        self.config = config
        self.stages = stages or build_default_stages()

    def dry_run_summary(self) -> dict[str, Any]:
        outputs = self.config.get("outputs", {})
        return {
            "dataset": self.config.get("dataset_building", {}).get("name"),
            "stage_order": [stage.name for stage in self.stages],
            "outputs": {stage.name: outputs.get(stage.output_key) for stage in self.stages},
            "status": "dataset_building_pending",
        }

    def run(self) -> None:
        raise NotImplementedError(
            "Dataset construction implementation is pending. "
            "Use --dry-run to validate the stage order and output paths."
        )
