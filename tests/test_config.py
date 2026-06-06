from pathlib import Path

from medimage_toolvqa_ms.config import load_yaml_config


def test_load_dataset_config():
    config = load_yaml_config(Path("configs/dataset_building.yaml"))

    assert config["dataset_building"]["name"] == "MedImage-ToolVQA"
    assert config["quality"]["require_tool_trace"] is True
