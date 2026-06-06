from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    records_path = Path(path)
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield value
