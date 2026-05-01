#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROMPT_TEMPLATE = """You are evaluating goal proximity for a robot manipulation task.

Task goal:
{task_meta}

You will be shown two images, A and B, from different demonstrations of the same task.
Choose the image that is closer to successfully completing the task goal.

Reply with exactly one token:
A
or
B
"""


README_TEMPLATE = """# LIBERO Pair Eval

Files:
- `pairs.jsonl`: pairwise evaluation samples
- `manifest.json`: dataset configuration and summary
- `images/`: exported frame images
- `run_openai_compatible_eval.py`: inference script for a local OpenAI-compatible vision endpoint

Quick start:
1. Upload this directory or the tarball to your cluster.
2. Start your local vision-language model with an OpenAI-compatible API.
3. Run:

```bash
python3 run_openai_compatible_eval.py \\
  --pairs pairs.jsonl \\
  --output predictions.jsonl \\
  --base-url http://127.0.0.1:8000/v1 \\
  --model YOUR_MODEL_NAME
```

The script writes per-sample predictions and prints accuracy if labels are present.
"""


INFERENCE_SCRIPT = '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib import request


PROMPT_TEMPLATE = """You are evaluating goal proximity for a robot manipulation task.

Task goal:
{task_meta}

You will be shown two images, A and B, from different demonstrations of the same task.
Choose the image that is closer to successfully completing the task goal.

Reply with exactly one token:
A
or
B
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO pair eval against a local OpenAI-compatible endpoint.")
    parser.add_argument("--pairs", type=Path, default=Path("pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def infer_one(args: argparse.Namespace, root: Path, sample: dict) -> str:
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE.format(task_meta=sample["task_meta"])},
                    {"type": "text", "text": "Image A"},
                    {"type": "image_url", "image_url": {"url": image_data_url(root / sample["image_a"])}},
                    {"type": "text", "text": "Image B"},
                    {"type": "image_url", "image_url": {"url": image_data_url(root / sample["image_b"])}},
                ],
            }
        ],
    }
    req = request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )
    with request.urlopen(req, timeout=args.timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"].strip()
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return text


def main() -> None:
    args = parse_args()
    root = args.pairs.parent
    samples = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total = 0

    with args.output.open("w", encoding="utf-8") as out_file:
        for sample in samples:
            prediction = infer_one(args, root, sample)
            record = {
                "pair_id": sample["pair_id"],
                "label": sample["label"],
                "prediction": prediction,
                "correct": prediction == sample["label"],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\\n")
            total += 1
            correct += int(record["correct"])

    if total:
        print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class FrameRef:
    demo: str
    frame: int
    tau: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small pairwise LIBERO evaluation set with exported images."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("raw_data/libero_spatial"))
    parser.add_argument("--output-dir", type=Path, default=Path("raw_data/libero_pair_eval_v1"))
    parser.add_argument("--pairs-per-task", type=int, default=100)
    parser.add_argument("--camera", default="agentview_rgb")
    parser.add_argument("--tau-min", type=float, default=0.10)
    parser.add_argument("--tau-max", type=float, default=0.90)
    parser.add_argument("--tau-window", type=float, default=0.10)
    parser.add_argument("--margin-quantile", type=float, default=0.60)
    parser.add_argument("--proposal-pool-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--make-tarball", action="store_true")
    return parser.parse_args()


def task_name_from_path(path: Path) -> str:
    return path.stem.removesuffix("_demo")


def task_meta_from_name(name: str) -> str:
    return name.replace("_", " ")


def tau_for_frame(frame_index: int, num_frames: int) -> float:
    if num_frames <= 1:
        return 0.0
    return frame_index / (num_frames - 1)


def nearest_frame_for_tau(tau: float, num_frames: int) -> int:
    return int(round(tau * (num_frames - 1)))


def build_score_trace(states: np.ndarray) -> np.ndarray:
    bowl_xyz = states[:, 10:13]
    goal_xyz = states[:, 38:41]
    return -np.linalg.norm(bowl_xyz - goal_xyz, axis=1)


def sample_frame_ref(
    rng: random.Random,
    demos: dict[str, dict[str, object]],
    tau_min: float,
    tau_max: float,
    tau_window: float,
) -> tuple[FrameRef, FrameRef]:
    demo_names = list(demos.keys())
    demo_a, demo_b = rng.sample(demo_names, 2)
    tau_a = rng.uniform(tau_min, tau_max)
    tau_b_min = max(tau_min, tau_a - tau_window)
    tau_b_max = min(tau_max, tau_a + tau_window)
    tau_b = rng.uniform(tau_b_min, tau_b_max)

    num_frames_a = int(demos[demo_a]["num_frames"])
    num_frames_b = int(demos[demo_b]["num_frames"])
    frame_a = nearest_frame_for_tau(tau_a, num_frames_a)
    frame_b = nearest_frame_for_tau(tau_b, num_frames_b)

    tau_a_actual = tau_for_frame(frame_a, num_frames_a)
    tau_b_actual = tau_for_frame(frame_b, num_frames_b)
    score_a = float(demos[demo_a]["scores"][frame_a])
    score_b = float(demos[demo_b]["scores"][frame_b])

    return (
        FrameRef(demo=demo_a, frame=frame_a, tau=tau_a_actual, score=score_a),
        FrameRef(demo=demo_b, frame=frame_b, tau=tau_b_actual, score=score_b),
    )


def export_image(
    task_dir: Path,
    task_name: str,
    camera: str,
    frame_ref: FrameRef,
    demo_group: h5py.Group,
) -> str:
    image_dir = task_dir / "images" / task_name
    image_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{frame_ref.demo}_{camera}_{frame_ref.frame:04d}.png"
    image_path = image_dir / image_name
    if not image_path.exists():
        image = Image.fromarray(np.asarray(demo_group["obs"][camera][frame_ref.frame]))
        image = image.transpose(Image.Transpose.ROTATE_180)
        image.save(image_path)
    return str(image_path.relative_to(task_dir))


def write_support_files(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    script_path = output_dir / "run_openai_compatible_eval.py"
    script_path.write_text(INFERENCE_SCRIPT, encoding="utf-8")
    script_path.chmod(0o755)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_support_files(output_dir)

    pairs_path = output_dir / "pairs.jsonl"
    manifest_path = output_dir / "manifest.json"
    task_summaries: list[dict[str, object]] = []

    with pairs_path.open("w", encoding="utf-8") as pairs_file:
        for task_index, h5_path in enumerate(sorted(args.input_dir.glob("*.hdf5")), start=1):
            task_name = task_name_from_path(h5_path)
            task_meta = task_meta_from_name(task_name)
            task_dir = output_dir

            with h5py.File(h5_path, "r") as h5_file:
                demos = {}
                for demo_name in sorted(h5_file["data"].keys(), key=lambda name: int(name.split("_")[-1])):
                    states = np.asarray(h5_file["data"][demo_name]["states"])
                    demos[demo_name] = {
                        "num_frames": states.shape[0],
                        "scores": build_score_trace(states),
                    }

                proposal_gaps = []
                for _ in range(args.proposal_pool_size):
                    frame_a, frame_b = sample_frame_ref(
                        rng=rng,
                        demos=demos,
                        tau_min=args.tau_min,
                        tau_max=args.tau_max,
                        tau_window=args.tau_window,
                    )
                    proposal_gaps.append(abs(frame_a.score - frame_b.score))
                margin = float(np.quantile(np.asarray(proposal_gaps), args.margin_quantile))

                seen: set[tuple[str, int, str, int]] = set()
                accepted = 0
                attempts = 0
                score_gaps: list[float] = []
                tau_gaps: list[float] = []

                while accepted < args.pairs_per_task:
                    attempts += 1
                    if attempts > args.pairs_per_task * 500:
                        raise RuntimeError(f"Could not sample enough pairs for {task_name}")

                    frame_a, frame_b = sample_frame_ref(
                        rng=rng,
                        demos=demos,
                        tau_min=args.tau_min,
                        tau_max=args.tau_max,
                        tau_window=args.tau_window,
                    )
                    score_gap = abs(frame_a.score - frame_b.score)
                    if score_gap < margin:
                        continue

                    canonical = tuple(sorted([(frame_a.demo, frame_a.frame), (frame_b.demo, frame_b.frame)]))
                    pair_key = (canonical[0][0], canonical[0][1], canonical[1][0], canonical[1][1])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    label = "A" if frame_a.score > frame_b.score else "B"
                    if rng.random() < 0.5:
                        first, second = frame_a, frame_b
                    else:
                        first, second = frame_b, frame_a
                        label = "A" if label == "B" else "B"

                    image_a = export_image(task_dir, task_name, args.camera, first, h5_file["data"][first.demo])
                    image_b = export_image(task_dir, task_name, args.camera, second, h5_file["data"][second.demo])

                    record = {
                        "pair_id": f"{task_index:02d}_{accepted + 1:04d}",
                        "task_name": task_name,
                        "task_meta": task_meta,
                        "source_file": h5_path.name,
                        "camera": args.camera,
                        "demo_a": first.demo,
                        "frame_a": first.frame,
                        "tau_a": round(first.tau, 6),
                        "score_a": round(first.score, 8),
                        "image_a": image_a,
                        "demo_b": second.demo,
                        "frame_b": second.frame,
                        "tau_b": round(second.tau, 6),
                        "score_b": round(second.score, 8),
                        "image_b": image_b,
                        "label": label,
                        "score_gap": round(abs(first.score - second.score), 8),
                        "tau_gap": round(abs(first.tau - second.tau), 6),
                    }
                    pairs_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted += 1
                    score_gaps.append(record["score_gap"])
                    tau_gaps.append(record["tau_gap"])

                task_summaries.append(
                    {
                        "task_name": task_name,
                        "source_file": h5_path.name,
                        "pairs": accepted,
                        "margin": margin,
                        "attempts": attempts,
                        "score_gap_mean": float(np.mean(score_gaps)),
                        "score_gap_min": float(np.min(score_gaps)),
                        "tau_gap_mean": float(np.mean(tau_gaps)),
                        "tau_gap_max": float(np.max(tau_gaps)),
                    }
                )

    manifest = {
        "dataset_name": output_dir.name,
        "num_tasks": len(task_summaries),
        "pairs_per_task": args.pairs_per_task,
        "total_pairs": args.pairs_per_task * len(task_summaries),
        "camera": args.camera,
        "tau_range": [args.tau_min, args.tau_max],
        "tau_window": args.tau_window,
        "score_proxy": "-L2(states[:,10:13], states[:,38:41])",
        "margin_quantile": args.margin_quantile,
        "seed": args.seed,
        "prompt_template": PROMPT_TEMPLATE,
        "tasks": task_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.make_tarball:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar_file:
            tar_file.add(output_dir, arcname=output_dir.name)
        print(f"Wrote tarball to {tar_path}")

    print(f"Wrote pairs to {pairs_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
