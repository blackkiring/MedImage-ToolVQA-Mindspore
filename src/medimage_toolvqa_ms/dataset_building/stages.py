from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    name: str
    input_key: str
    output_key: str
    description: str


def build_default_stages() -> tuple[PipelineStage, ...]:
    return (
        PipelineStage(
            name="merge",
            input_key="region_sources",
            output_key="merged_regions",
            description="Merge region-level results, bbox, mask, target text, and source metadata.",
        ),
        PipelineStage(
            name="make_vqa",
            input_key="merged_regions",
            output_key="vqa_candidates",
            description="Create image-dependent VQA candidates without leaking bbox or mask hints.",
        ),
        PipelineStage(
            name="verify",
            input_key="vqa_candidates",
            output_key="verified_samples",
            description="Check answer consistency, visual grounding, tool arguments, and source attribution.",
        ),
        PipelineStage(
            name="makereasoning",
            input_key="verified_samples",
            output_key="tool_trajectories",
            description="Synthesize tool-use trajectories and observation references.",
        ),
        PipelineStage(
            name="make_sft",
            input_key="tool_trajectories",
            output_key="sft_records",
            description="Convert trajectories into multi-turn SFT records.",
        ),
    )
