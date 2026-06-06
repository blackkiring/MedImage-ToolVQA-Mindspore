from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from medimage_toolvqa_ms.data.dataset import iter_jsonl_records
from medimage_toolvqa_ms.dataset_building.stages import (
    PipelineStage,
    build_default_stages,
    make_sft,
    make_vqa,
    makereasoning,
    merge_regions,
    verify_samples,
    write_jsonl,
)

logger = logging.getLogger(__name__)


class DatasetBuildPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        stages: tuple[PipelineStage, ...] | None = None,
    ) -> None:
        self.config = config
        self.stages = stages or build_default_stages()

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------

    def dry_run_summary(self) -> dict[str, Any]:
        outputs = self.config.get("outputs", {})
        inputs = self.config.get("inputs", {})

        input_counts: dict[str, int] = {}
        for key, path in inputs.items():
            if isinstance(path, list):
                total = 0
                for p in path:
                    if Path(p).exists():
                        total += sum(1 for _ in iter_jsonl_records(p))
                input_counts[key] = total
            elif isinstance(path, str) and Path(path).exists():
                input_counts[key] = sum(1 for _ in iter_jsonl_records(path))

        output_counts: dict[str, int] = {}
        for key, path in outputs.items():
            if isinstance(path, str) and Path(path).exists():
                output_counts[key] = sum(1 for _ in iter_jsonl_records(path))

        return {
            "dataset": self.config.get("dataset_building", {}).get("name"),
            "stage_order": [stage.name for stage in self.stages],
            "outputs": {
                stage.name: outputs.get(stage.output_key) for stage in self.stages
            },
            "input_counts": input_counts,
            "output_counts": output_counts,
            "status": (
                "dataset_building_complete"
                if output_counts
                else "dataset_building_pending"
            ),
        }

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def _resolve_llm_client(self, llm_config: dict[str, Any]) -> Any:
        provider = llm_config.get("provider", "mock")
        if provider == "mock":
            try:
                from medimage_toolvqa_ms.llm import MockLLMClient

                return MockLLMClient()
            except ImportError:
                return None
        elif provider == "http":
            try:
                from medimage_toolvqa_ms.llm import HttpLLMClient

                return HttpLLMClient(
                    base_url=llm_config.get("base_url", "http://localhost:8000"),
                    api_key=llm_config.get("api_key", ""),
                    model=llm_config.get("model", ""),
                )
            except ImportError:
                return None
        return None

    def run(self) -> None:
        inputs = self.config.get("inputs", {})
        outputs = self.config.get("outputs", {})
        llm_config = self.config.get("llm", {})
        dataset_cfg = self.config.get("dataset_building", {})

        llm_client = self._resolve_llm_client(llm_config)

        print(f"[pipeline] Starting dataset building: {dataset_cfg.get('name', 'unknown')}")

        # Stage 1: merge
        print("[pipeline] Stage 1/5: merge_regions")
        region_sources = inputs.get("region_sources", [])
        source_records = inputs.get("source_records", "")
        merged = merge_regions(region_sources, source_records, self.config)
        n = write_jsonl(outputs["merged_regions"], merged)
        print(f"  -> {n} merged regions")

        # Stage 2: make_vqa
        print("[pipeline] Stage 2/5: make_vqa")
        vqa = make_vqa(merged, llm_client, self.config)
        n = write_jsonl(outputs["vqa_candidates"], vqa)
        print(f"  -> {n} VQA candidates")

        # Stage 3: verify
        print("[pipeline] Stage 3/5: verify_samples")
        verified, needs_review = verify_samples(vqa, self.config)
        n = write_jsonl(outputs["verified_samples"], verified)
        review_path = outputs["verified_samples"].replace(".jsonl", "_review.jsonl")
        if needs_review:
            write_jsonl(review_path, needs_review)
        print(f"  -> {n} verified, {len(needs_review)} need review")

        # Stage 4: makereasoning
        print("[pipeline] Stage 4/5: makereasoning")
        trajectories = makereasoning(verified, llm_client, self.config)
        n = write_jsonl(outputs["tool_trajectories"], trajectories)
        print(f"  -> {n} tool trajectories")

        # Stage 5: make_sft
        print("[pipeline] Stage 5/5: make_sft")
        sft_records = make_sft(trajectories, self.config)
        n = write_jsonl(outputs["sft_records"], sft_records)
        print(f"  -> {n} SFT records")

        print("[pipeline] Dataset building complete.")
