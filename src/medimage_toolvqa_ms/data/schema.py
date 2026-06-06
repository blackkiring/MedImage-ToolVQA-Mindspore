from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_RECORD_KEYS = {
    "sample_id",
    "task_type",
    "image_context",
    "diagnosis_schema",
    "visual_evidence",
    "messages",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


def validate_sft_record(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    missing = sorted(REQUIRED_RECORD_KEYS - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    messages = record.get("messages")
    if messages is not None and not isinstance(messages, list):
        errors.append("messages must be a list")

    evidence = record.get("visual_evidence")
    if isinstance(evidence, dict):
        bbox = evidence.get("bbox_2d")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
            errors.append("visual_evidence.bbox_2d must contain four numbers")

    return ValidationResult(ok=not errors, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Pipeline record types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergedRegion:
    sample_id: str
    image_path: str
    mask_path: str | None
    bbox_2d: list[float]
    target_description: str
    source_dataset: str
    source_url: str
    license: str
    modality: str
    anatomical_region: str


@dataclass(frozen=True)
class VQACandidate:
    sample_id: str
    image_path: str
    question: str
    options: list[str]
    answer: str
    question_type: str
    source_dataset: str
    visual_evidence_ref: str
    generation_model: str
    generation_params: dict[str, Any]


@dataclass(frozen=True)
class VerifiedSample:
    sample_id: str
    image_path: str
    mask_path: str | None
    bbox_2d: list[float] | None
    target_description: str
    question: str
    options: list[str]
    answer: str
    question_type: str
    source_dataset: str
    modality: str
    anatomical_region: str
    needs_human_review: bool
    review_reason: str | None


@dataclass(frozen=True)
class ToolStep:
    tool_name: str
    tool_input: dict[str, Any]
    observation_image: str | None
    reasoning: str


@dataclass(frozen=True)
class ToolTrajectory:
    sample_id: str
    image_path: str
    mask_path: str | None
    bbox_2d: list[float] | None
    target_description: str
    question: str
    options: list[str]
    answer: str
    question_type: str
    source_dataset: str
    modality: str
    anatomical_region: str
    tool_steps: list[ToolStep]
    final_reasoning: str
    observation_images: list[str]


@dataclass(frozen=True)
class SFTMessage:
    role: str
    content: str
    images: list[str] | None = None


@dataclass(frozen=True)
class SFTRecord:
    sample_id: str
    source_dataset: str
    question: str
    answer: str
    question_type: str
    messages: list[SFTMessage]
    images: list[str]
    tool_trace_length: int
    score: float


# ---------------------------------------------------------------------------
# ROI-leak detection words
# ---------------------------------------------------------------------------

_BBOX_LEAK_WORDS = frozenset({
    "bounding box", "bbox", "mask", "roi", "region of interest",
    "marked region", "boxed area", "marked area",
})


# ---------------------------------------------------------------------------
# Stage-specific validation helpers
# ---------------------------------------------------------------------------

def validate_merged_region(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    required = {"sample_id", "image_path", "bbox_2d", "target_description",
                "source_dataset", "source_url", "license", "modality", "anatomical_region"}
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    bbox = record.get("bbox_2d")
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
        errors.append("bbox_2d must contain four numbers")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_vqa_candidate(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    required = {"sample_id", "image_path", "question", "options", "answer", "question_type"}
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    answer = record.get("answer", "")
    if not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCDE":
        errors.append("answer must be a single letter A-E")
    options = record.get("options", [])
    if not isinstance(options, list) or len(options) < 4:
        errors.append("options must have at least 4 entries")
    question = record.get("question", "").lower()
    for word in _BBOX_LEAK_WORDS:
        if word in question:
            errors.append(f"question contains ROI-leak word: {word!r}")
            break
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_verified_sample(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    required = {"sample_id", "image_path", "question", "options", "answer",
                "question_type", "source_dataset", "modality", "anatomical_region"}
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    bbox = record.get("bbox_2d")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append("bbox_2d must contain four numbers")
        elif all(isinstance(v, (int, float)) for v in bbox):
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                errors.append("bbox_2d must satisfy x1<x2 and y1<y2")
            if any(v < 0 for v in bbox):
                errors.append("bbox_2d coordinates must be non-negative")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_tool_trajectory(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    from medimage_toolvqa_ms.tools.tool_schema import allowed_tool_names
    allowed = allowed_tool_names()
    steps = record.get("tool_steps", [])
    if not isinstance(steps, list):
        errors.append("tool_steps must be a list")
    else:
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"tool_steps[{i}] must be a dict")
                continue
            tn = step.get("tool_name", "")
            if tn not in allowed:
                errors.append(f"tool_steps[{i}].tool_name {tn!r} not in allowed tools")
    required = {"sample_id", "image_path", "question", "answer", "tool_steps",
                "final_reasoning", "observation_images"}
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_sft_record_enhanced(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    required = {"sample_id", "source_dataset", "question", "answer",
                "messages", "images"}
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    messages = record.get("messages")
    if isinstance(messages, list):
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}] must be a dict")
                continue
            if msg.get("role") not in ("system", "user", "assistant"):
                errors.append(f"messages[{i}].role must be system/user/assistant")
            if not isinstance(msg.get("content", ""), str):
                errors.append(f"messages[{i}].content must be a string")
    elif messages is not None:
        errors.append("messages must be a list")
    return ValidationResult(ok=not errors, errors=tuple(errors))
