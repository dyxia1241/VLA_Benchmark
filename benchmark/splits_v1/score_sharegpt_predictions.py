#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_JSON = THIS_DIR / "sharegpt_export_v2" / "test_sharegpt.json"

ANSWER_TAG_RE = re.compile(r"<\s*ANSWER\s*>(.*?)<\s*/\s*ANSWER\s*>", flags=re.IGNORECASE | re.DOTALL)
OPTION_LINE_RE = re.compile(r"^\s*([A-Z]+)\.\s+(.*?)\s*$")

PRED_CANDIDATE_KEYS = (
    "predict",
    "prediction",
    "pred",
    "predict_text",
    "generated_text",
    "output",
    "response",
    "text",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score LLaMA-Factory predictions against ShareGPT-exported benchmark test set."
    )
    p.add_argument("--test-json", default=str(DEFAULT_TEST_JSON))
    p.add_argument("--predictions", required=True, help="Prediction file from LLaMA-Factory.")
    p.add_argument(
        "--pred-key",
        default="",
        help="If predictions are JSON objects, force this field as the raw prediction text.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Optional path to save a machine-readable summary JSON.",
    )
    p.add_argument(
        "--dump-jsonl",
        default="",
        help="Optional path to save per-example scored rows.",
    )
    return p.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_choices_from_prompt(prompt: str) -> dict[str, str]:
    lines = str(prompt).splitlines()
    in_options = False
    choices: dict[str, str] = {}
    for line in lines:
        line = line.rstrip()
        if normalize_space(line).lower() == "options:":
            in_options = True
            continue
        if not in_options:
            continue
        m = OPTION_LINE_RE.match(line)
        if m:
            key = m.group(1).upper()
            val = normalize_space(m.group(2))
            choices[key] = val
            continue
        if normalize_space(line):
            break
    return choices


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(data):
        convs = ex.get("conversations", [])
        prompt = str(convs[0].get("value", "")) if len(convs) >= 1 else ""
        target = str(convs[1].get("value", "")) if len(convs) >= 2 else ""
        choices = extract_choices_from_prompt(prompt)
        rows.append(
            {
                "index": idx,
                "task_type": str(ex.get("task_type", "")),
                "dataset": str(ex.get("source_dataset") or ex.get("dataset") or ""),
                "task_id": str(ex.get("task_id", "")),
                "recording_id": str(ex.get("recording_id", "")),
                "answer": str(ex.get("answer", "")).strip().upper(),
                "target_text": target,
                "prompt": prompt,
                "choices": choices,
            }
        )
    return rows


def pick_prediction_text(obj: Any, pred_key: str) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported prediction record type: {type(obj).__name__}")
    if pred_key:
        if pred_key not in obj:
            raise KeyError(f"pred-key={pred_key!r} not found in prediction record keys={sorted(obj.keys())}")
        return str(obj[pred_key])
    for key in PRED_CANDIDATE_KEYS:
        if key in obj:
            return str(obj[key])
    string_keys = [k for k, v in obj.items() if isinstance(v, str)]
    if len(string_keys) == 1:
        return str(obj[string_keys[0]])
    raise KeyError(
        "Could not infer prediction text field. "
        f"Available keys={sorted(obj.keys())}. Pass --pred-key explicitly."
    )


def load_predictions(path: Path, pred_key: str) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    rows: list[str] = []

    if suffixes.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(line.rstrip("\n"))
                    continue
                rows.append(pick_prediction_text(obj, pred_key))
        return rows

    if suffixes.endswith(".json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise TypeError("JSON predictions file must contain a list.")
        for item in obj:
            rows.append(pick_prediction_text(item, pred_key))
        return rows

    if suffixes.endswith(".txt"):
        with path.open("r", encoding="utf-8") as fh:
            rows = [line.rstrip("\n") for line in fh if line.strip()]
        return rows

    raise ValueError(f"Unsupported prediction file format: {path}")


def extract_tag_payload(text: str) -> str:
    matches = list(ANSWER_TAG_RE.finditer(str(text)))
    if not matches:
        return ""
    return str(matches[-1].group(1)).strip()


def parse_from_valid_set(text: str, valid_set: set[str]) -> str:
    if not valid_set:
        return "INVALID"

    text_u = str(text).strip().upper().replace("。", ".")
    alt = "|".join(sorted((re.escape(v) for v in valid_set), key=len, reverse=True))

    text_compact = re.sub(r"[\s\.\:\-\(\)\[\]\{\}<>/]+", "", text_u)
    if text_compact in valid_set:
        return text_compact

    cue_pat = rf"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE|FINAL)\s*(?:IS|:|=)?\s*\(?({alt})\)?(?![A-Z0-9])"
    cue_matches = list(re.finditer(cue_pat, text_u))
    if cue_matches:
        return cue_matches[-1].group(1)

    tok_pat = rf"(?<![A-Z0-9])({alt})(?![A-Z0-9])"
    tok_matches = list(re.finditer(tok_pat, text_u))
    if tok_matches:
        return tok_matches[-1].group(1)

    return "INVALID"


def parse_temporal_text(text: str, labels: list[str]) -> str:
    text_u = str(text).strip().upper().replace("。", ".")
    label_set = set(labels)
    perms = {"".join(p) for p in itertools.permutations(labels, len(labels))}
    chars = "".join(sorted(label_set))

    exact_hits = []
    for m in re.finditer(rf"(?<![A-Z])([{chars}]{{{len(labels)}}})(?![A-Z])", text_u):
        cand = m.group(1)
        if cand in perms:
            exact_hits.append(cand)
    if exact_hits:
        return exact_hits[-1]

    sep_hits = []
    sep_pat = rf"(?<![A-Z])([{chars}])(?:\s*[,>\-\/|]+\s*|\s+)([{chars}])(?:\s*[,>\-\/|]+\s*|\s+)([{chars}])(?![A-Z])"
    for m in re.finditer(sep_pat, text_u):
        cand = "".join(m.groups())
        if cand in perms:
            sep_hits.append(cand)
    if sep_hits:
        return sep_hits[-1]

    standalone = re.findall(rf"(?<![A-Z])([{chars}])(?![A-Z])", text_u)
    if len(standalone) >= len(labels):
        window_hits = []
        for i in range(len(standalone) - len(labels) + 1):
            cand = "".join(standalone[i : i + len(labels)])
            if cand in perms:
                window_hits.append(cand)
        if window_hits:
            return window_hits[-1]

    filtered = "".join(ch for ch in text_u if ch in label_set)
    if len(filtered) >= len(labels):
        tail = filtered[-len(labels) :]
        if tail in perms:
            return tail
        head = filtered[: len(labels)]
        if head in perms:
            return head

    return "INVALID"


def parse_prediction(raw: str, gold: dict[str, Any]) -> str:
    answer = str(gold.get("answer", "")).strip().upper()
    choices: dict[str, str] = gold.get("choices", {}) or {}
    valid_set = set(choices.keys())
    tagged = extract_tag_payload(raw)

    is_temporal = (
        gold.get("task_type") == "T_temporal"
        or (len(answer) == 3 and set(answer).issubset({"X", "Y", "Z"}))
    )
    if is_temporal:
        labels = sorted(set(answer))
        if tagged:
            pred = parse_temporal_text(tagged, labels)
            if pred != "INVALID":
                return pred
        return parse_temporal_text(raw, labels)

    if valid_set:
        if tagged:
            pred = parse_from_valid_set(tagged, valid_set)
            if pred != "INVALID":
                return pred

        pred = parse_from_valid_set(raw, valid_set)
        if pred != "INVALID":
            return pred

        raw_norm = re.sub(r"\s+", " ", str(raw).upper()).strip()
        hits: list[tuple[int, str]] = []
        for key, value in choices.items():
            kk = str(key).upper()
            vv = re.sub(r"\s+", " ", str(value).upper()).strip()
            if not vv:
                continue
            pos = raw_norm.find(vv)
            if pos >= 0:
                hits.append((pos, kk))
        if hits:
            hits.sort(key=lambda x: x[0])
            return hits[-1][1]

    # Very last fallback: accept raw answer if it exactly matches the gold label.
    raw_compact = re.sub(r"[\s\.\:\-\(\)\[\]\{\}<>/]+", "", str(raw).upper())
    if raw_compact == answer:
        return answer
    if tagged:
        tagged_compact = re.sub(r"[\s\.\:\-\(\)\[\]\{\}<>/]+", "", tagged.upper())
        if tagged_compact == answer:
            return answer

    return "INVALID"


def pct(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    args = parse_args()
    gold_rows = load_gold_rows(Path(args.test_json))
    pred_rows = load_predictions(Path(args.predictions), args.pred_key)

    if len(pred_rows) != len(gold_rows):
        raise ValueError(
            f"Prediction count mismatch: got {len(pred_rows)} predictions, "
            f"but test set has {len(gold_rows)} items."
        )

    by_type: dict[str, list[bool]] = defaultdict(list)
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    invalid_by_type: dict[str, int] = defaultdict(int)
    scored_rows: list[dict[str, Any]] = []

    for gold, raw_pred in zip(gold_rows, pred_rows):
        parsed = parse_prediction(raw_pred, gold)
        correct = parsed == gold["answer"]
        by_type[gold["task_type"]].append(correct)
        by_dataset[gold["dataset"]].append(correct)
        if parsed == "INVALID":
            invalid_by_type[gold["task_type"]] += 1
        scored_rows.append(
            {
                "index": gold["index"],
                "dataset": gold["dataset"],
                "task_type": gold["task_type"],
                "task_id": gold["task_id"],
                "recording_id": gold["recording_id"],
                "gold_answer": gold["answer"],
                "pred_answer": parsed,
                "raw_prediction": raw_pred,
                "correct": correct,
            }
        )

    overall = sum(r["correct"] for r in scored_rows) / len(scored_rows) if scored_rows else 0.0

    print("\nBy Task Type")
    print(f"{'Task':<14} {'Acc':>8} {'N':>8} {'Invalid':>8}")
    print("-" * 42)
    task_summary: dict[str, Any] = {}
    for task_type in sorted(by_type):
        vals = by_type[task_type]
        acc = sum(vals) / len(vals) if vals else 0.0
        invalid = invalid_by_type.get(task_type, 0)
        task_summary[task_type] = {"acc": acc, "n": len(vals), "invalid": invalid}
        print(f"{task_type:<14} {pct(acc):>8} {len(vals):>8} {invalid:>8}")

    print("\nBy Dataset")
    print(f"{'Dataset':<14} {'Acc':>8} {'N':>8}")
    print("-" * 32)
    dataset_summary: dict[str, Any] = {}
    for dataset in sorted(by_dataset):
        vals = by_dataset[dataset]
        acc = sum(vals) / len(vals) if vals else 0.0
        dataset_summary[dataset] = {"acc": acc, "n": len(vals)}
        print(f"{dataset:<14} {pct(acc):>8} {len(vals):>8}")

    total_invalid = sum(invalid_by_type.values())
    print("\nOverall")
    print(f"acc={pct(overall)}  n={len(scored_rows)}  invalid={total_invalid}")

    summary = {
        "test_json": str(Path(args.test_json).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "pred_key": args.pred_key,
        "overall_acc": overall,
        "n_items": len(scored_rows),
        "n_invalid": total_invalid,
        "by_task_type": task_summary,
        "by_dataset": dataset_summary,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dump_jsonl:
        out_path = Path(args.dump_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in scored_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
