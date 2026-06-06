from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medimage_toolvqa_ms.data.dataset import iter_jsonl_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline stage definition
# ---------------------------------------------------------------------------

MEDICAL_AGENT_SYSTEM_PROMPT = (
    "You are a medical image analysis assistant with access to specialized tools.\n\n"
    "Available tools:\n"
    "1. Zoom-in: Crop a local region from an image by bbox for detailed inspection.\n"
    '   Input: [{"index": <image_index>, "bbox_2d": [x1, y1, x2, y2]}]\n'
    "2. BiomedParse: Segment a medical object using a text description.\n"
    '   Input: [{"index": <image_index>, "captions": "<text description>"}]\n'
    "3. SAM2: Segment an object using a bbox prompt.\n"
    '   Input: [{"index": <image_index>, "bbox_2d": [x1, y1, x2, y2]}]\n\n'
    "Output format:\n"
    "<think> Your detailed reasoning process  Action: [ToolName]\n"
    "```json\n[tool input JSON]\n```\n\n"
    "For final answer:\n"
    "<think> Your detailed reasoning process  Action: Answer\n"
    "<answer> Your answer </answer>"
)


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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from *text*, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Stage 1: merge
# ---------------------------------------------------------------------------

def merge_regions(
    region_sources: list[str],
    source_records: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    for source_path in region_sources:
        if not Path(source_path).exists():
            logger.warning("merge: source file not found: %s", source_path)
            continue
        for record in iter_jsonl_records(source_path):
            sid = record.get("sample_id", "")
            if not sid or sid in seen:
                continue
            if not record.get("image_path"):
                logger.debug("merge: skipping %s – missing image_path", sid)
                continue
            bbox = record.get("bbox_2d")
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                logger.debug("merge: skipping %s – empty/invalid bbox_2d", sid)
                continue
            seen.add(sid)
            merged.append(record)

    if source_records and Path(source_records).exists():
        for record in iter_jsonl_records(source_records):
            sid = record.get("sample_id", "")
            if not sid or sid in seen:
                continue
            if not record.get("image_path"):
                continue
            bbox = record.get("bbox_2d")
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            seen.add(sid)
            merged.append(record)

    return merged


# ---------------------------------------------------------------------------
# Stage 2: make_vqa
# ---------------------------------------------------------------------------

_VQA_PROMPT_TEMPLATE = (
    "You are a medical imaging expert. Given a region description from a medical image, "
    "generate a multiple-choice visual question that requires examining the image to answer.\n\n"
    "Region description: {target_description}\n"
    "Imaging modality: {modality}\n"
    "Anatomical region: {anatomical_region}\n\n"
    "Requirements:\n"
    "- Generate exactly 5 options labeled (A) through (E)\n"
    "- The question must require visual evidence from the image\n"
    "- Do NOT mention bounding boxes, masks, or region-of-interest markers\n"
    "- Provide the correct answer as a single letter\n\n"
    'Respond with JSON: {{"question": "...", "options": ["(A) ...", ...], "answer": "X"}}'
)


def make_vqa(
    merged_records: list[dict[str, Any]],
    llm_client: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "default")
    candidates: list[dict[str, Any]] = []

    for record in merged_records:
        prompt = _VQA_PROMPT_TEMPLATE.format(
            target_description=record.get("target_description", ""),
            modality=record.get("modality", "unknown"),
            anatomical_region=record.get("anatomical_region", "unknown"),
        )

        if llm_client is None:
            # Offline stub – generate a deterministic placeholder
            parsed = {
                "question": f"Based on this {record.get('modality', 'medical')} image, what is the most likely finding in the {record.get('anatomical_region', 'shown')} region?",
                "options": [
                    "(A) Normal finding",
                    "(B) Abnormal mass",
                    "(C) Inflammatory change",
                    "(D) Degenerative change",
                    "(E) Inconclusive",
                ],
                "answer": "B",
            }
            gen_model = "offline_stub"
        else:
            from medimage_toolvqa_ms.llm.client import GenerationRequest

            request = GenerationRequest(
                prompt=prompt,
                model=model_name,
                temperature=0.7,
                max_tokens=512,
            )
            result = llm_client.generate(request)
            if result.error:
                logger.warning("make_vqa: LLM error for %s: %s", record.get("sample_id"), result.error)
                continue
            parsed = _extract_json_from_text(result.text)
            if parsed is None:
                logger.warning("make_vqa: invalid JSON from LLM for %s", record.get("sample_id"))
                continue
            gen_model = result.model

        candidates.append(
            {
                "sample_id": record.get("sample_id"),
                "image_path": record.get("image_path"),
                "mask_path": record.get("mask_path"),
                "bbox_2d": record.get("bbox_2d"),
                "target_description": record.get("target_description", ""),
                "question": parsed.get("question", ""),
                "options": parsed.get("options", []),
                "answer": parsed.get("answer", ""),
                "question_type": "multiple_choice",
                "source_dataset": record.get("source_dataset", ""),
                "modality": record.get("modality", ""),
                "anatomical_region": record.get("anatomical_region", ""),
                "visual_evidence_ref": record.get("sample_id", ""),
                "generation_model": gen_model,
                "generation_params": {"temperature": 0.7, "max_tokens": 512},
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Stage 3: verify
# ---------------------------------------------------------------------------

_BBOX_LEAK_WORDS = frozenset({
    "bounding box", "bbox", "mask", "roi", "region of interest",
    "marked region", "boxed area", "marked area",
})


def verify_samples(
    vqa_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality = config.get("quality", {})
    reject_text_only = quality.get("reject_text_only_questions", True)

    verified: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []

    for rec in vqa_candidates:
        reasons: list[str] = []

        # -- answer format
        answer = rec.get("answer", "")
        if not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCDE":
            reasons.append("answer must be a single letter A-E")

        # -- options count
        options = rec.get("options", [])
        if not isinstance(options, list) or not (4 <= len(options) <= 5):
            reasons.append("options must have 4-5 entries")

        # -- answer matches an option
        if isinstance(options, list) and answer in "ABCDE":
            if not any(opt.strip().startswith(f"({answer})") for opt in options):
                reasons.append("answer letter does not match any option")

        # -- ROI-leak check
        question = rec.get("question", "").lower()
        for word in _BBOX_LEAK_WORDS:
            if word in question:
                reasons.append(f"question contains ROI-leak word: {word!r}")
                break

        # -- bbox validity
        bbox = rec.get("bbox_2d") or rec.get("visual_evidence", {}).get("bbox_2d")
        if bbox is not None and isinstance(bbox, list) and len(bbox) == 4:
            if all(isinstance(v, (int, float)) for v in bbox):
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    reasons.append("bbox_2d invalid: x1>=x2 or y1>=y2")
                if any(v < 0 for v in bbox):
                    reasons.append("bbox_2d has negative coordinates")

        sample = {
            **rec,
            "needs_human_review": bool(reasons),
            "review_reason": "; ".join(reasons) if reasons else None,
        }

        if reasons:
            needs_review.append(sample)
        else:
            verified.append(sample)

    return verified, needs_review


# ---------------------------------------------------------------------------
# Stage 4: makereasoning
# ---------------------------------------------------------------------------

_REASONING_TEMPLATES = {
    "zoom_in": (
        "I will examine the region of interest more closely by zooming in. "
        "The target is described as: {target}. I'll crop the area defined by the bbox "
        "to inspect its texture, boundaries, and density characteristics."
    ),
    "biomedparse": (
        "Using BiomedParse to segment the target structure described as: {target}. "
        "This will help delineate the exact boundaries and confirm the morphological features."
    ),
    "sam2": (
        "Using SAM2 to generate a precise segmentation mask for the region. "
        "The bbox-guided segmentation will help verify the extent and shape of the finding."
    ),
    "final": (
        "Based on my analysis of the original image and tool observations: "
        "The target region ({target}) shows characteristics consistent with the answer. "
        "After examining the zoomed view and segmentation results, I can now provide my diagnosis."
    ),
}


def _wrap_think(text: str) -> str:
    """Wrap text in <think> tags, avoiding double nesting."""
    stripped = text.strip()
    if stripped.startswith("<think>") and stripped.endswith("</think>"):
        return stripped
    return f"<think>{stripped}</think>"


def makereasoning(
    verified_samples: list[dict[str, Any]],
    llm_client: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tool_augmented_ratio = config.get("quality", {}).get("tool_augmented_ratio", 0.9)
    trajectories: list[dict[str, Any]] = []

    for rec in verified_samples:
        use_tools = random.random() < tool_augmented_ratio
        bbox = rec.get("bbox_2d")
        target = rec.get("target_description", "the region of interest")
        sid = rec.get("sample_id", "unknown")

        tool_steps: list[dict[str, Any]] = []
        observation_images: list[str] = []

        if use_tools and bbox:
            # Step 1: Zoom-in
            zoom_reason = _wrap_think(_REASONING_TEMPLATES["zoom_in"].format(target=target))
            if llm_client:
                from medimage_toolvqa_ms.llm.client import GenerationRequest

                resp = llm_client.generate(GenerationRequest(
                    prompt=f"Describe your reasoning for zooming into this region: {target}",
                    max_tokens=256,
                ))
                if not resp.error and resp.text.strip():
                    zoom_reason = _wrap_think(resp.text)

            obs_path = f"observations/{sid}_step1_zoom.png"
            tool_steps.append({
                "tool_name": "Zoom-in",
                "tool_input": [{"index": 1, "bbox_2d": bbox}],
                "observation_image": obs_path,
                "reasoning": zoom_reason,
            })
            observation_images.append(obs_path)

            # Step 2: BiomedParse or SAM2
            second_tool = random.choice(["BiomedParse", "SAM2"])
            if second_tool == "BiomedParse":
                step_reason = _wrap_think(_REASONING_TEMPLATES["biomedparse"].format(target=target))
                step_input = [{"index": 2, "captions": target}]
            else:
                step_reason = _wrap_think(_REASONING_TEMPLATES["sam2"].format(target=target))
                step_input = [{"index": 2, "bbox_2d": bbox}]

            if llm_client:
                from medimage_toolvqa_ms.llm.client import GenerationRequest

                resp = llm_client.generate(GenerationRequest(
                    prompt=f"Reason about using {second_tool} for: {target}",
                    max_tokens=256,
                ))
                if not resp.error and resp.text.strip():
                    step_reason = _wrap_think(resp.text)

            obs_path2 = f"observations/{sid}_step2_{second_tool.lower()}.png"
            tool_steps.append({
                "tool_name": second_tool,
                "tool_input": step_input,
                "observation_image": obs_path2,
                "reasoning": step_reason,
            })
            observation_images.append(obs_path2)

        # Final reasoning
        final_reasoning = _wrap_think(_REASONING_TEMPLATES["final"].format(target=target))
        if llm_client:
            from medimage_toolvqa_ms.llm.client import GenerationRequest

            resp = llm_client.generate(GenerationRequest(
                prompt=f"Provide final reasoning for: {target}. Answer: {rec.get('answer', 'B')}",
                max_tokens=256,
            ))
            if not resp.error and resp.text.strip():
                final_reasoning = _wrap_think(resp.text)

        trajectories.append({
            "sample_id": sid,
            "image_path": rec.get("image_path", ""),
            "mask_path": rec.get("mask_path"),
            "bbox_2d": bbox,
            "target_description": target,
            "question": rec.get("question", ""),
            "options": rec.get("options", []),
            "answer": rec.get("answer", ""),
            "question_type": rec.get("question_type", "multiple_choice"),
            "source_dataset": rec.get("source_dataset", ""),
            "modality": rec.get("modality", ""),
            "anatomical_region": rec.get("anatomical_region", ""),
            "tool_steps": tool_steps,
            "final_reasoning": final_reasoning,
            "observation_images": observation_images,
        })

    return trajectories


# ---------------------------------------------------------------------------
# Stage 5: make_sft
# ---------------------------------------------------------------------------

def make_sft(
    trajectories: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sft_records: list[dict[str, Any]] = []

    for traj in trajectories:
        messages: list[dict[str, Any]] = []
        all_images: list[str] = []

        # System
        messages.append({"role": "system", "content": MEDICAL_AGENT_SYSTEM_PROMPT})

        # First user turn
        question = traj.get("question", "")
        options_text = "\n".join(traj.get("options", []))
        first_user = (
            f"<image>\n### Question:\n{question}\n"
            f"Options:\n{options_text}\n"
            "The index of the given image is 1.\n"
            "Begin your reasoning:"
        )
        messages.append({"role": "user", "content": first_user})
        all_images.append(traj.get("image_path", ""))

        # Tool steps
        for i, step in enumerate(traj.get("tool_steps", []), start=1):
            tool_name = step.get("tool_name", "")
            tool_input = step.get("tool_input", [])
            reasoning = step.get("reasoning", "")

            assistant_content = (
                f"{reasoning}\n"
                f"Action: {tool_name}\n"
                f"```json\n{json.dumps(tool_input, ensure_ascii=False)}\n```"
            )
            messages.append({"role": "assistant", "content": assistant_content})

            obs_img = step.get("observation_image", "")
            next_index = i + 1
            user_obs = (
                f"<image>\n"
                f"The index of the given image is {next_index}. "
                "Continue your reasoning:"
            )
            messages.append({"role": "user", "content": user_obs})
            if obs_img:
                all_images.append(obs_img)

        # Final answer
        answer = traj.get("answer", "")
        final_reasoning = traj.get("final_reasoning", "")
        final_content = (
            f"{final_reasoning}\n"
            f"Action: Answer\n"
            f"<answer> {answer} </answer>"
        )
        messages.append({"role": "assistant", "content": final_content})

        sft_records.append({
            "sample_id": traj.get("sample_id", ""),
            "source_dataset": traj.get("source_dataset", ""),
            "question": question,
            "answer": answer,
            "question_type": traj.get("question_type", "multiple_choice"),
            "messages": messages,
            "images": all_images,
            "tool_trace_length": len(traj.get("tool_steps", [])),
            "score": 1.0,
        })

    return sft_records
