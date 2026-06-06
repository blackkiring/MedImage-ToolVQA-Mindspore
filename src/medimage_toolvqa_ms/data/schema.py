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
