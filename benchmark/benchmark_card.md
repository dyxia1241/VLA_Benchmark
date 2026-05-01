---
pretty_name: "ProcessBench"
license: "mit"
tags:
  - robotics
  - benchmark
  - vision-language-models
  - manipulation
  - process-understanding
  - multimodal
  - video
  - image
  - mlcroissant
task_categories:
  - visual-question-answering
task_ids:
  - visual-question-answering
language:
  - en
size_categories:
  - 10K<n<100K
configs:
  - config_name: splits_v1
    data_files:
      - split: train
        path: "splits_v1/all_sft_merged.jsonl"
      - split: test
        path: "splits_v1/all_eval_merged.jsonl"
---

# ProcessBench: Benchmark Card (Current Four-source Mainline)

## 1. Overview

**Name:** ProcessBench
**Status:** current repository mainline draft, aligned to `splits_v1`
**Positioning:** a goal-conditioned, signal-grounded, process-aware benchmark for robotic manipulation
**Benchmark root:** `/data/projects/GM-100/benchmark`
**Current mainline sources:** `GM-100`, `RH20T`, `REASSEMBLE`, `AIST`
**Current mainline scope:** four-source benchmark construction, strict `SFT / Eval` split, and one-click evaluation/aggregation
**Not part of the minimal mainline:** `RoboMIND2.0` remains an appendix-scale future extension, not a required source for the current paper closure

ProcessBench is designed to evaluate **robotic manipulation process understanding** rather than generic visual recognition. The benchmark focuses on whether a model can infer:

- the current process state
- whether effective contact has been established
- the dominant motion direction
- how far a local action has progressed
- whether an operation is moving, succeeding, or failing
- what primitive is happening now, what comes next, and how a primitive chain should be restored

The benchmark is intentionally **multi-source** and **task-heterogeneous**. It does not require every source dataset to support every task. Instead, each dataset contributes only to the tasks that are stable under its native signals.

## 2. Evaluative Role

### 2.1 Primary claim

ProcessBench supports the following claim:

> VLMs differ systematically in their ability to understand robotic manipulation processes, and those differences can be measured through structured tasks grounded in dataset-native signals across heterogeneous robot datasets.

### 2.2 What the benchmark is intended to measure

- process phase recognition
- contact state recognition
- motion direction prediction
- bimanual coordination state recognition
- primitive-local progress understanding
- motion-state recognition
- early outcome prediction
- temporal ordering and temporal priority reasoning
- current primitive, next primitive, and primitive-chain reasoning

### 2.3 What the benchmark does not claim

- closed-loop VLA policy quality by itself
- deployment or safety guarantees
- broad cross-robot generalization without additional experiments
- generic all-purpose VLM ranking validity

## 3. Benchmark Topology

### 3.1 Two task families

| Family | Goal | Tasks |
| --- | --- | --- |
| Static process monitoring | Recognize current state, contact, coordination, or current primitive. | `T1 / T2 / T4 / T10` |
| Dynamic process reasoning | Predict direction, progress, outcome, temporal relations, or next-step structure. | `T3 / T5 / T6 / T7 / T8 / T9 / T11 / T12` |

Naming convention:

- `T5 == T_progress`
- `T8 == T_temporal`
- `T9 == T_binary`

Paper-facing text should use `T5 / T8 / T9`. Engineering files may keep the legacy names for compatibility.

### 3.2 Task inventory

| ID | Name | Core question | Input form | Main sources |
| --- | --- | --- | --- | --- |
| `T1` | Phase Recognition | Which coarse manipulation phase is the episode currently in? | single frame | GM-100, RH20T, REASSEMBLE |
| `T2` | Contact Detection | Has effective contact already happened? | single frame | GM-100, RH20T, REASSEMBLE |
| `T3` | Motion Direction Prediction | What is the dominant motion direction? | short ordered frame context | GM-100, RH20T, AIST |
| `T4` | Bimanual Coordination State | How are the two arms coordinating right now? | short ordered frame context | GM-100, AIST |
| `T5` | Primitive-local Progress | Is the current local step in early / middle / late progress? | short ordered frame context | GM-100, RH20T, REASSEMBLE |
| `T6` | Motion State Recognition | Is the scene currently in a clear motion state? | short ordered frame context | GM-100, RH20T, REASSEMBLE, AIST |
| `T7` | Operation Outcome Prediction | Given an early clip, will the operation eventually succeed or fail? | longer-gap multi-frame context | RH20T, REASSEMBLE |
| `T8` | Temporal Ordering | What is the true order of three shuffled observations? | 3 shuffled frames | GM-100, RH20T, REASSEMBLE, AIST |
| `T9` | Temporal Priority Prediction | Which of two observations happened earlier? | 2-panel / 2-frame comparison | GM-100, RH20T, REASSEMBLE, AIST |
| `T10` | Current Primitive Recognition | Which low-level primitive is happening now? | short ordered frame context | REASSEMBLE |
| `T11` | Next Primitive Prediction | What primitive should happen next? | short context + current process state | REASSEMBLE |
| `T12` | Primitive Chain Restoration | How should a local primitive chain be restored? | partial chain / masked chain | REASSEMBLE |

## 4. Dataset Roles and Scale

### 4.1 Source roles

| Dataset | Role | Supported tasks | Current frozen status |
| --- | --- | --- | --- |
| `GM-100` | Main goal-conditioned source; covers both single-arm and bimanual settings | `T1, T2, T3, T4, T5, T6, T8, T9` | `15,500` items |
| `RH20T` | Process/contact source driven by force/torque, gripper, and TCP signals | `T1, T2, T3, T5, T6, T7, T8, T9` | `15,800` items, `596` scenes, `76` task IDs |
| `REASSEMBLE` | Cleanest source for action-chain, primitive, and outcome reasoning | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` | `17,165` items, `37` recordings |
| `AIST-Bimanual` | Bimanual kinematics source without force signal | `T3, T4, T6, T8, T9` | `9,427` items, `200` episodes |

Coverage rule:

- not every dataset must support all `12` tasks
- each dataset only supports tasks that are stable under its native signals

### 4.2 Current full benchmark scale

| Source | Items | Coverage |
| --- | ---: | --- |
| `GM-100` | `15,500` | `T1, T2, T3, T4, T5, T6, T8, T9` |
| `RH20T` | `15,800` | `T1, T2, T3, T5, T6, T7, T8, T9` |
| `REASSEMBLE` | `17,165` | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` |
| `AIST` | `9,427` | `T3, T4, T6, T8, T9` |

Current full benchmark total: **`57,892` items**.

### 4.3 Canonical artifacts in the current mainline

| Source | Curated JSONL | Frame cache | Dataset-specific eval wrapper |
| --- | --- | --- | --- |
| GM-100 | `benchmark/benchmark_v1_curated.jsonl` | `benchmark/benchmark_v1_frames_tbinary_20260330/` | `benchmark/eval_v1/run_benchmark_v1_eval.py` |
| RH20T | `benchmark/rh20t_benchmark_v0_curated.jsonl` | `benchmark/rh20t_benchmark_v0_frames/` | `benchmark/eval_v1/run_rh20t_benchmark_v0_eval.py` |
| REASSEMBLE | `benchmark/reassemble_benchmark_v0_curated.jsonl` | `benchmark/reassemble_benchmark_v0_frames/` | `benchmark/eval_v1/run_reassemble_benchmark_v0_eval.py` |
| AIST | `benchmark/aist_benchmark_v0/aist_benchmark_v0_curated.jsonl` | `benchmark/aist_benchmark_v0_frames/` | `benchmark/eval_v1/run_aist_benchmark_v0_eval.py` |

## 5. Split Protocol

### 5.1 Current default split

The current mainline uses a strict `episode-level` isolation rule:

- `85%` SFT
- `15%` Eval
- the same `episode / recording / scene` appears in exactly one split

Canonical split artifacts:

- `benchmark/splits_v1/all_sft_merged.jsonl`
- `benchmark/splits_v1/all_eval_merged.jsonl`
- `benchmark/splits_v1/split_summary.json`
- `benchmark/splits_v1/eval_manifest.json`

### 5.2 Current split counts

| Source | Total | Eval | SFT | Group unit | Eval groups |
| --- | ---: | ---: | ---: | --- | ---: |
| `GM-100` | `15,500` | `2,643` | `12,857` | `(task_id, episode_id)` | `628` |
| `RH20T` | `15,800` | `2,422` | `13,378` | `recording_id` | `89` |
| `REASSEMBLE` | `17,165` | `2,562` | `14,603` | `recording_id` | `6` |
| `AIST` | `9,427` | `1,424` | `8,003` | `recording_id` | `30` |

Merged totals:

- Eval: **`9,051`**
- SFT: **`48,841`**

### 5.3 Split design note

`REASSEMBLE` is the only source that carries `T10 / T11 / T12`, and it has only `37` recordings. The current `splits_v1` design therefore uses a weighted `recording-level` selection for `REASSEMBLE`, so those three fragile task types are not diluted away in eval.

## 6. Ground-truth Construction

### 6.1 GT principles

ProcessBench follows five main GT rules:

1. Prefer dataset-native signals such as force/torque, gripper state, TCP, joint state, velocity, official segment labels, and task metadata.
2. Define the local unit for each task first, then sample frames within that unit.
3. Build GT automatically; reserve human work for audit and taxonomy calibration.
4. Treat `task_meta` as a controllable variable and report clearly whether it is enabled during evaluation.
5. Keep `primitive-object pair` annotations as auxiliary analysis metadata rather than promoting them to the benchmark’s unified main GT.

### 6.2 Shared primitive closed set

The shared primitive ontology uses coarse process primitives:

| Primitive | Meaning |
| --- | --- |
| `engage` | establish effective contact, grasp, hold, or take over the object |
| `stabilize` | keep, press, support, or fix the object |
| `transport` | move, push, pull, or transfer object position |
| `align` | align, locate, adjust pose, or pre-insertion registration |
| `effect` | commit the state-changing action |
| `disengage` | release, withdraw, or end the interaction |
| `failure` | explicit deviation from the goal, drop, non-completion, or wrong contact |

`reach / approach` is intentionally excluded from the shared closed set because it is usually a pre-contact transition with low information gain and high annotation noise.

### 6.3 Minimal manual annotation scope

Human annotation is intentionally restricted to:

- task-level primitive chains
- object slots

The benchmark does **not** require:

- dense frame-level primitive annotation
- dense object grounding annotation
- verb-object labels as the unified benchmark GT

### 6.4 `T5` definition

`T5` is defined as **within-local-step progress**.

Continuous GT:

`u = (t - start(P_k)) / (end(P_k) - start(P_k))`

Released labels:

- `early`
- `middle`
- `late`

Local-step source by dataset:

| Dataset | Local-step source |
| --- | --- |
| `GM-100` | contact-aware local interaction interval |
| `RH20T` | local interval synthesized from force/torque + gripper + TCP |
| `REASSEMBLE` | official successful high-level segment |

## 7. Evaluation Protocol

### 7.1 Default mainline protocol

The recommended protocol is **not** to collapse all four sources into one giant JSONL with a shared frame root. Instead:

1. keep dataset-specific eval JSONL files
2. keep dataset-specific frame caches
3. run a unified controller that dispatches dataset-specific evaluation
4. aggregate results afterward

This design is the current default because the four sources use different frame caches, while the shared execution path `benchmark/eval_v1/run_pilot_eval.py` only accepts one frame-root configuration at a time.

### 7.2 One-click mainline entrypoints

Rebuild the split:

```bash
cd /data/projects/GM-100
python3 benchmark/gt_build/build_multisource_sft_eval_split.py
```

Run all eval sets:

```bash
cd /data/projects/GM-100
python3 benchmark/eval_v1/run_all_eval_sets.py --model <model> --api-key <key>
```

Aggregate scores:

```bash
cd /data/projects/GM-100
python3 benchmark/eval_v1/score_all_eval_sets.py --model <model>
```

### 7.3 Minimal runtime closure

If the goal is only to run the current mainline eval and aggregate results, the minimal closure depends on:

- `benchmark/eval_v1/run_all_eval_sets.py`
- `benchmark/eval_v1/run_pilot_eval.py`
- `benchmark/eval_v1/score_all_eval_sets.py`
- `benchmark/splits_v1/eval_manifest.json`
- `benchmark/splits_v1/gm100_eval.jsonl`
- `benchmark/splits_v1/rh20t_eval.jsonl`
- `benchmark/splits_v1/reassemble_eval.jsonl`
- `benchmark/splits_v1/aist_eval.jsonl`
- `benchmark/benchmark_v1_frames_tbinary_20260330/`
- `benchmark/rh20t_benchmark_v0_frames/`
- `benchmark/reassemble_benchmark_v0_frames/`
- `benchmark/aist_benchmark_v0_frames/`

### 7.4 Default outputs

The default output directory is:

- `benchmark/eval_results_v1/splits_v1/<model_slug>/`

Expected outputs:

- `gm100_eval_results.jsonl`
- `rh20t_eval_results.jsonl`
- `reassemble_eval_results.jsonl`
- `aist_eval_results.jsonl`
- `run_all_eval_sets_summary.json`
- `score_all_eval_sets_summary.json`

### 7.5 Legacy-compatible path

`benchmark/run_v1_pipeline.sh` is retained only for the historical GM-100 single-source release path. It should be treated as **legacy-compatible**, not as part of the current four-source runtime-core.

## 8. Reliability and Manual Audit

### 8.1 Current audit policy

Human audit is currently reserved for:

1. benchmark validity audit
2. task-level primitive-object chain audit

The current primary annotator recorded in the repository is **Huiting Ji**.

### 8.2 High-priority audit targets

The current highest-priority audit targets are:

- `T5`
- `T7`
- `T10`
- `T11`
- `T12`
- `T3` direction mapping calibration

### 8.3 Current reliability boundary

The benchmark GT itself is builder-generated. Formal audit statistics are still an active work item for the current paper cycle, especially for the most fragile task families. Reliability claims in the paper should therefore distinguish clearly between:

- benchmark construction logic already frozen in code
- targeted audit evidence already available
- audit statistics that are still being completed

## 9. Limitations and Caveats

The current mainline has several explicit caveats:

1. `REASSEMBLE` has only `37` recordings, so `T10 / T11 / T12` remain statistically fragile compared with larger task families.
2. `AIST` has no force signal, so it supports bimanual kinematics and temporal reasoning, but not contact or failure as primary evidence.
3. `RH20T` uses a fixed main-camera protocol in the current mainline; the viewpoint choice should be stated explicitly as a caveat in paper reporting.
4. `RoboMIND2.0` is not yet part of the minimal paper closure and should not be written as if it were already a finished mainline result source.
5. Evaluation reports should state clearly whether `task_meta` prompting is enabled, because it is a controllable variable rather than part of the benchmark GT.

## 10. Intended Use

- diagnostic evaluation of VLM process understanding for robotic manipulation
- structured comparison across heterogeneous robot datasets
- per-task and per-source failure analysis
- training/evaluation protocol construction using `all_sft_merged.jsonl` and `all_eval_merged.jsonl`
- benchmarking of process-aware reasoning tasks rather than only static perception tasks

## 11. Out-of-Scope Use

- claiming causal VLA control improvement from benchmark score alone
- using one overall score as a deployment or safety certificate
- claiming cross-robot generalization without dedicated held-out transfer experiments
- using ProcessBench as a generic leaderboard for all-purpose VLM capability

## 12. Current Release Status

The current mainline is already stable enough to support the paper’s minimal closure:

1. define the process-aware benchmark topology
2. build a four-source benchmark
3. release strict `episode-level` `SFT / Eval` splits
4. provide one-click evaluation and aggregation

Current immediate next steps in the repository plan are:

- align all public-facing docs and the benchmark card to `splits_v1`
- run the four-source baseline and produce `score_all_eval_sets_summary.json`
- prepare single-model SFT from `all_sft_merged.jsonl`
- complete audit statistics for high-risk task types

## 13. Citation

```bibtex
@misc{processbench_2026,
  title        = {ProcessBench: A Goal-Conditioned, Signal-Grounded, Process-Aware Benchmark for Robotic Manipulation},
  author       = {TBD},
  year         = {2026},
  howpublished = {\url{TBD}},
  note         = {Current four-source mainline draft}
}
```

## 14. Contact

Maintainer: TBD
Repository issues: TBD
