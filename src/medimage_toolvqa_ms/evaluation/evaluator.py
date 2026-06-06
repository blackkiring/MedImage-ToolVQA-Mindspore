from __future__ import annotations

from typing import Any


class Evaluator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def dry_run_summary(self) -> dict[str, Any]:
        evaluation = self.config.get("evaluation", {})
        return {
            "records": evaluation.get("records"),
            "checkpoint": evaluation.get("checkpoint"),
            "output_file": evaluation.get("output_file"),
            "metrics": evaluation.get("metrics", []),
            "status": "evaluation_pending",
        }

    def run(self) -> None:
        raise NotImplementedError(
            "MindSpore evaluation implementation is pending. "
            "Use --dry-run to validate configuration loading."
        )
