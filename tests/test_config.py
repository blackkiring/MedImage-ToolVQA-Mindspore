from pathlib import Path

from medimage_toolvqa_ms.config import load_yaml_config


def test_load_dataset_config():
    config = load_yaml_config(Path("configs/dataset.yaml"))

    assert config["dataset"]["name"] == "MedImage-ToolVQA"
    assert config["schema"]["require_tool_trace"] is True
