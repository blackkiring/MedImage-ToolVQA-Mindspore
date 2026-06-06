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
