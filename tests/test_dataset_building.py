from medimage_toolvqa_ms.dataset_building import DatasetBuildPipeline, build_default_stages


def test_default_dataset_building_stage_order():
    assert [stage.name for stage in build_default_stages()] == [
        "merge",
        "make_vqa",
        "verify",
        "makereasoning",
        "make_sft",
    ]


def test_dataset_building_dry_run_summary():
    config = {
        "dataset_building": {"name": "MedImage-ToolVQA"},
        "outputs": {
            "merged_regions": "01.jsonl",
            "vqa_candidates": "02.jsonl",
            "verified_samples": "03.jsonl",
            "tool_trajectories": "04.jsonl",
            "sft_records": "05.jsonl",
        },
    }

    summary = DatasetBuildPipeline(config).dry_run_summary()

    assert summary["dataset"] == "MedImage-ToolVQA"
    assert summary["outputs"]["make_sft"] == "05.jsonl"


# ---------------------------------------------------------------------------
# Stage function tests
# ---------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path


def _write_jsonl(records):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
    return f.name


def test_merge_regions_deduplicates():
    from medimage_toolvqa_ms.dataset_building.stages import merge_regions

    records = [
        {"sample_id": "s1", "image_path": "img.png", "bbox_2d": [0, 0, 10, 10], "target_description": "x", "source_dataset": "BioMedParse", "source_url": "", "license": "", "modality": "CT", "anatomical_region": "chest"},
        {"sample_id": "s1", "image_path": "img2.png", "bbox_2d": [1, 1, 11, 11], "target_description": "y", "source_dataset": "BioMedParse", "source_url": "", "license": "", "modality": "CT", "anatomical_region": "chest"},
        {"sample_id": "s2", "image_path": "img3.png", "bbox_2d": [2, 2, 12, 12], "target_description": "z", "source_dataset": "BioMedParse", "source_url": "", "license": "", "modality": "CT", "anatomical_region": "chest"},
    ]
    path = _write_jsonl(records)
    result = merge_regions([path], "", {})
    assert len(result) == 2
    assert result[0]["sample_id"] == "s1"
    assert result[1]["sample_id"] == "s2"
    Path(path).unlink()


def test_merge_regions_skips_missing_fields():
    from medimage_toolvqa_ms.dataset_building.stages import merge_regions

    records = [
        {"sample_id": "s1"},  # missing image_path and bbox_2d
        {"sample_id": "s2", "image_path": "img.png", "bbox_2d": [0, 0, 10, 10], "target_description": "x", "source_dataset": "BioMedParse", "source_url": "", "license": "", "modality": "CT", "anatomical_region": "chest"},
    ]
    path = _write_jsonl(records)
    result = merge_regions([path], "", {})
    assert len(result) == 1
    assert result[0]["sample_id"] == "s2"
    Path(path).unlink()


def test_verify_samples_separates():
    from medimage_toolvqa_ms.dataset_building.stages import verify_samples

    candidates = [
        {"sample_id": "s1", "image_path": "img.png", "question": "What is shown?", "options": ["(A) Normal", "(B) Abnormal", "(C) Inconclusive", "(D) Benign", "(E) Malignant"], "answer": "B", "question_type": "multiple_choice", "source_dataset": "BioMedParse", "visual_evidence_ref": "", "generation_model": "", "generation_params": {}},
        {"sample_id": "s2", "image_path": "img.png", "question": "Look at the bounding box region", "options": ["(A) X", "(B) Y", "(C) Z", "(D) W"], "answer": "A", "question_type": "multiple_choice", "source_dataset": "BioMedParse", "visual_evidence_ref": "", "generation_model": "", "generation_params": {}},
    ]
    verified, needs_review = verify_samples(candidates, {})
    assert len(verified) == 1
    assert len(needs_review) == 1
    assert needs_review[0]["sample_id"] == "s2"


def test_verify_samples_checks_answer_letter():
    from medimage_toolvqa_ms.dataset_building.stages import verify_samples

    candidates = [
        {"sample_id": "s1", "image_path": "img.png", "question": "What?", "options": ["(A) X", "(B) Y", "(C) Z", "(D) W"], "answer": "Z", "question_type": "multiple_choice", "source_dataset": "BioMedParse", "visual_evidence_ref": "", "generation_model": "", "generation_params": {}},
    ]
    verified, needs_review = verify_samples(candidates, {})
    assert len(needs_review) == 1  # "Z" not in A-E


def test_makereasoning_creates_trajectory():
    from medimage_toolvqa_ms.dataset_building.stages import makereasoning

    samples = [
        {"sample_id": "s1", "image_path": "img.png", "mask_path": None, "bbox_2d": [100, 100, 300, 300], "target_description": "a mass", "question": "What?", "options": ["(A) X", "(B) Y", "(C) Z", "(D) W"], "answer": "B", "question_type": "multiple_choice", "source_dataset": "BioMedParse", "modality": "CT", "anatomical_region": "chest"},
    ]
    result = makereasoning(samples, None, {"quality": {"tool_augmented_ratio": 1.0}})
    assert len(result) == 1
    assert len(result[0]["tool_steps"]) >= 1
    assert result[0]["tool_steps"][0]["tool_name"] == "Zoom-in"


def test_make_sft_creates_messages():
    from medimage_toolvqa_ms.dataset_building.stages import make_sft

    trajectories = [
        {
            "sample_id": "s1", "image_path": "img.png", "mask_path": None,
            "bbox_2d": [100, 100, 300, 300], "target_description": "a mass",
            "question": "What is shown?", "options": ["(A) Normal", "(B) Abnormal", "(C) Inconclusive", "(D) Benign"],
            "answer": "B", "question_type": "multiple_choice", "source_dataset": "BioMedParse",
            "modality": "CT", "anatomical_region": "chest",
            "tool_steps": [
                {"tool_name": "Zoom-in", "tool_input": [{"index": 1, "bbox_2d": [100, 100, 300, 300]}], "observation_image": "obs1.png", "reasoning": "<think>Examining the region.</think>"},
            ],
            "final_reasoning": "<think>Based on the zoomed view, this is abnormal.</think>",
            "observation_images": ["obs1.png"],
        },
    ]
    result = make_sft(trajectories, {})
    assert len(result) == 1
    msgs = result[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"  # tool call
    assert msgs[3]["role"] == "user"  # observation
    assert msgs[4]["role"] == "assistant"  # final answer
    assert "<answer>" in msgs[4]["content"]
    assert len(result[0]["images"]) == 2  # original + observation


def test_make_sft_no_tools():
    from medimage_toolvqa_ms.dataset_building.stages import make_sft

    trajectories = [
        {
            "sample_id": "s1", "image_path": "img.png", "mask_path": None,
            "bbox_2d": [100, 100, 300, 300], "target_description": "a mass",
            "question": "What?", "options": ["(A) X", "(B) Y", "(C) Z", "(D) W"],
            "answer": "A", "question_type": "multiple_choice", "source_dataset": "BioMedParse",
            "modality": "CT", "anatomical_region": "chest",
            "tool_steps": [],
            "final_reasoning": "<think>Direct reasoning.</think>",
            "observation_images": [],
        },
    ]
    result = make_sft(trajectories, {})
    assert len(result) == 1
    msgs = result[0]["messages"]
    # system + user + final assistant = 3 messages
    assert len(msgs) == 3
    assert result[0]["tool_trace_length"] == 0
