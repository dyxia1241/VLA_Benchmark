#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_aist_pilot_cards import write_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render AIST T3 pilot audit cards.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(input_jsonl)
    manifest_path = output_dir / "cards_manifest.jsonl"
    cards = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, item in enumerate(items, start=1):
            card_path = write_card(output_dir, item, frame_dir, idx)
            row = {
                "item_index": idx,
                "task_type": item.get("task_type", ""),
                "task_id": item.get("task_id", ""),
                "recording_id": item.get("recording_id", ""),
                "card_path": str(card_path),
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            cards.append(row)

    summary = {
        "input_jsonl": str(input_jsonl),
        "frame_dir": str(frame_dir),
        "output_dir": str(output_dir),
        "num_items": len(items),
        "num_cards": len(cards),
    }
    (output_dir / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
