#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether LIBERO state vectors expose object-level goal proximity proxies."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_data/libero_spatial"),
        help="Directory containing LIBERO spatial .hdf5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("raw_data/libero_spatial_extracted/state_proxy_analysis.json"),
        help="Path to write analysis JSON.",
    )
    return parser.parse_args()


def sorted_demos(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def detect_pose_blocks(states: np.ndarray) -> list[int]:
    blocks: list[int] = []
    for start in range(10, states.shape[1] - 6):
        quat_norm = np.linalg.norm(states[:, start + 3 : start + 7], axis=1)
        if np.mean(np.abs(quat_norm - 1.0)) < 1e-6:
            if not blocks or start - blocks[-1] >= 7:
                blocks.append(start)
    return blocks


def mean_corr_with_time(distance_traces: list[np.ndarray]) -> float:
    correlations: list[float] = []
    for trace in distance_traces:
        tau = np.linspace(0.0, 1.0, len(trace))
        if np.std(trace) < 1e-9:
            continue
        correlations.append(float(np.corrcoef(tau, -trace)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


def improvement_fraction(distance_traces: list[np.ndarray]) -> float:
    improved = sum(float(trace[-1] < trace[0]) for trace in distance_traces)
    return improved / max(len(distance_traces), 1)


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"No .hdf5 files found in {args.input_dir}")

    per_file: list[dict[str, object]] = []
    global_pose_blocks: list[int] | None = None

    for path in files:
        with h5py.File(path, "r") as h5_file:
            demos = sorted_demos(h5_file["data"])
            demo_states = [np.asarray(h5_file["data"][demo]["states"]) for demo in demos]
            concat_states = np.concatenate(demo_states, axis=0)
            pose_blocks = detect_pose_blocks(concat_states)
            if global_pose_blocks is None:
                global_pose_blocks = pose_blocks

            movement_by_block: dict[int, float] = {}
            for block in pose_blocks:
                movement = [
                    float(np.linalg.norm(states[-1, block : block + 3] - states[0, block : block + 3]))
                    for states in demo_states
                ]
                movement_by_block[block] = float(np.mean(movement))

            movable_block = max(movement_by_block, key=movement_by_block.get)

            final_movable_positions = np.stack([states[-1, movable_block : movable_block + 3] for states in demo_states])
            candidate_goal_blocks = [block for block in pose_blocks if block != movable_block]
            goal_distances = {}
            for block in candidate_goal_blocks:
                candidate_positions = np.stack([states[-1, block : block + 3] for states in demo_states])
                goal_distances[block] = float(
                    np.linalg.norm(candidate_positions.mean(axis=0) - final_movable_positions.mean(axis=0))
                )
            goal_block = min(goal_distances, key=goal_distances.get)

            traces = [
                np.linalg.norm(
                    states[:, movable_block : movable_block + 3] - states[:, goal_block : goal_block + 3],
                    axis=1,
                )
                for states in demo_states
            ]

            file_result = {
                "file": path.name,
                "pose_blocks": pose_blocks,
                "movable_block": movable_block,
                "goal_block": goal_block,
                "mean_block_motion": {str(key): value for key, value in movement_by_block.items()},
                "goal_distance_to_terminal_mean": {str(key): value for key, value in goal_distances.items()},
                "distance_stats": {
                    "initial_mean": float(np.mean([trace[0] for trace in traces])),
                    "final_mean": float(np.mean([trace[-1] for trace in traces])),
                    "improvement_fraction": improvement_fraction(traces),
                    "mean_corr_tau_with_neg_distance": mean_corr_with_time(traces),
                },
                "mean_initial_xyz_by_block": {
                    str(block): np.mean([states[0, block : block + 3] for states in demo_states], axis=0).round(6).tolist()
                    for block in pose_blocks
                },
                "mean_final_xyz_by_block": {
                    str(block): np.mean([states[-1, block : block + 3] for states in demo_states], axis=0).round(6).tolist()
                    for block in pose_blocks
                },
            }
            per_file.append(file_result)

    summary = {
        "state_dim": int(concat_states.shape[1]),
        "pose_blocks": global_pose_blocks,
        "interpretation": {
            "movable_block": "Most likely the manipulated black bowl pose block.",
            "goal_block": "Most likely the plate target pose block.",
        },
        "files": per_file,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote analysis to {args.output}")


if __name__ == "__main__":
    main()
