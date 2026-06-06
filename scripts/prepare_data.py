#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medimage_toolvqa_ms.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MedImage-ToolVQA records.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only print config summary.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0

    raise NotImplementedError("Data preparation implementation is pending.")


if __name__ == "__main__":
    raise SystemExit(main())
