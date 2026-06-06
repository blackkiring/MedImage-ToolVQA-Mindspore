#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medimage_toolvqa_ms.config import load_yaml_config
from medimage_toolvqa_ms.dataset_building import DatasetBuildPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MedImage-ToolVQA dataset records.")
    parser.add_argument("--config", required=True, help="Path to dataset construction YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only print stage summary.")
    args = parser.parse_args()

    pipeline = DatasetBuildPipeline(load_yaml_config(args.config))
    if args.dry_run:
        print(json.dumps(pipeline.dry_run_summary(), ensure_ascii=False, indent=2))
        return 0

    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
