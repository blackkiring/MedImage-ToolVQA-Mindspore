#!/usr/bin/env python
"""将外部 BioMedParse SFT 数据转换为本项目 pipeline 格式。

用法:
    python scripts/convert_external_sft.py \
        --input /fs-ift/med/xulin-ustc/LLaMA-Factory/data/agent_sft_train_split.json \
        --output data/processed/05_sft_records.jsonl \
        --path-prefix "/vepfs-vpc-mlp2" \
        --also-regions data/raw/regions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# 图片路径前缀替换
_DEFAULT_STRIP_PREFIX = "/vepfs-vpc-mlp2"


def _fix_image_path(path: str, strip_prefix: str) -> str:
    if strip_prefix and path.startswith(strip_prefix):
        return path[len(strip_prefix):]
    return path


def _extract_tool_calls(content: str) -> list[dict]:
    """从 assistant content 中提取工具调用。"""
    pattern = r" Action:\s*<tool_call>(\w[\w-]*)\s*\n```(?:json)?\s*\n(.*?)\n```</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)
    calls = []
    for tool_name, json_str in matches:
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            payload = []
        calls.append({"tool": tool_name, "payload": payload})
    return calls


def _extract_answer(content: str) -> str | None:
    m = re.search(r"<answer>\s*([A-Ea-e])\s*</answer>", content)
    if m:
        return m.group(1).upper()
    return None


def _extract_bbox_from_messages(messages: list[dict]) -> list[float] | None:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        calls = _extract_tool_calls(msg.get("content", ""))
        for call in calls:
            payload = call.get("payload", [])
            if isinstance(payload, dict):
                payload = [payload]
            for item in payload:
                bbox = item.get("bbox_2d")
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    return bbox
    return None


def convert_record(rec: dict, strip_prefix: str) -> dict:
    """转换单条外部 SFT 记录为本项目格式（保留原始 <think> 标签）。"""
    images = [_fix_image_path(img, strip_prefix) for img in rec.get("images", [])]

    messages = []
    for msg in rec.get("messages", []):
        messages.append({
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        })

    return {
        "sample_id": f"ext_{hash(rec.get('question', '') + rec.get('answer', '')) & 0xFFFFFFFF:08x}",
        "source_dataset": rec.get("source", "BioMedParse"),
        "question": rec.get("question", ""),
        "answer": rec.get("answer", ""),
        "question_type": rec.get("question_type", "multiple_choice"),
        "messages": messages,
        "images": images,
        "tool_trace_length": max(0, len([m for m in messages if m.get("role") == "assistant"]) - 1),
        "score": rec.get("score", 1.0),
        "lang": rec.get("lang", "en"),
    }


def extract_region_from_record(rec: dict, strip_prefix: str) -> dict | None:
    """从外部 SFT 记录中提取区域级信息（用于 merge 阶段）。"""
    images = rec.get("images", [])
    if not images:
        return None

    bbox = _extract_bbox_from_messages(rec.get("messages", []))
    question = rec.get("question", "")
    answer = rec.get("answer", "")

    return {
        "sample_id": f"ext_{hash(question + answer) & 0xFFFFFFFF:08x}",
        "image_path": _fix_image_path(images[0], strip_prefix),
        "mask_path": None,
        "bbox_2d": bbox or [0, 0, 512, 512],
        "target_description": question[:100],
        "source_dataset": rec.get("source", "BioMedParse"),
        "source_url": "",
        "license": "",
        "modality": "",
        "anatomical_region": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="转换外部 BioMedParse SFT 数据")
    parser.add_argument("--input", required=True, help="外部 SFT JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 SFT records JSONL 路径")
    parser.add_argument("--path-prefix", default=_DEFAULT_STRIP_PREFIX, help="要去掉的路径前缀")
    parser.add_argument("--also-regions", default=None, help="同时输出 regions.jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="最多转换条数")
    args = parser.parse_args()

    print(f"加载外部数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]
    print(f"共 {len(data)} 条记录")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sft_count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in data:
            converted = convert_record(rec, args.path_prefix)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            sft_count += 1
    print(f"写入 SFT records: {args.output} ({sft_count} 条)")

    if args.also_regions:
        Path(args.also_regions).parent.mkdir(parents=True, exist_ok=True)
        region_count = 0
        seen = set()
        with open(args.also_regions, "w", encoding="utf-8") as f:
            for rec in data:
                region = extract_region_from_record(rec, args.path_prefix)
                if region and region["sample_id"] not in seen:
                    seen.add(region["sample_id"])
                    f.write(json.dumps(region, ensure_ascii=False) + "\n")
                    region_count += 1
        print(f"写入 regions: {args.also_regions} ({region_count} 条)")

    print("转换完成。")


if __name__ == "__main__":
    main()
