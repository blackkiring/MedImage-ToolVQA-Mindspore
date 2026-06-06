from medimage_toolvqa_ms.data.schema import validate_sft_record
from medimage_toolvqa_ms.tools.tool_schema import allowed_tool_names


def test_validate_minimal_sft_record():
    record = {
        "sample_id": "sample_001",
        "task_type": "medical_image_vqa_with_tool_use",
        "image_context": {"modality": "X-ray"},
        "diagnosis_schema": {"not_for_clinical_diagnosis": True},
        "visual_evidence": {"bbox_2d": [1, 2, 3, 4]},
        "messages": [],
    }

    result = validate_sft_record(record)

    assert result.ok is True
    assert result.errors == ()


def test_allowed_tool_names():
    assert allowed_tool_names() == {"Zoom-in", "BiomedParse", "SAM2"}
