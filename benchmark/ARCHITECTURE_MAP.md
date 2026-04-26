# Benchmark Architecture Map

更新时间：2026-04-19  
适用范围：`/data/projects/GM-100/benchmark`

定位说明：

- 本文件记录当前稳定主线的目录结构、数据流和默认入口。
- `NEURIPS_ED_MASTER_PLAN.md` 负责记录论文主线、数据源角色、split 策略和投稿边界。
- `README.md` 只保留最高频的运行方式。

## 1. Current Scope

当前稳定主线已经收敛到四个数据源：

- `GM-100`
- `RH20T`
- `REASSEMBLE`
- `AIST`

当前冻结全量题库总数：`57,892`

当前冻结 split：

- `SFT`: `48,841`
- `Eval`: `9,051`

## 2. Directory Layout

```text
GM-100/
├── gm100-cobotmagic-lerobot/                        # GM-100 原始数据
├── reassemble-tuwien-researchdata/                 # REASSEMBLE 官方 recording 目录
├── aist-bimanip/                                   # AIST 选定 task 与 selected20 episodes
├── benchmark/
│   ├── GM100 List.xlsx
│   ├── benchmark_card.md
│   ├── NEURIPS_ED_MASTER_PLAN.md
│   ├── ARCHITECTURE_MAP.md
│   ├── eval_v1/
│   │   ├── run_pilot_eval.py
│   │   ├── run_benchmark_v1_eval.py
│   │   ├── run_rh20t_benchmark_v0_eval.py
│   │   ├── run_reassemble_benchmark_v0_eval.py
│   │   ├── run_aist_benchmark_v0_eval.py
│   │   ├── run_all_eval_sets.py
│   │   ├── score_pilot.py
│   │   ├── score_benchmark_v1.py
│   │   ├── score_rh20t_benchmark_v0.py
│   │   ├── score_reassemble_benchmark_v0.py
│   │   ├── score_aist_benchmark_v0.py
│   │   └── score_all_eval_sets.py
│   ├── gt_build/
│   │   ├── build_sampling_pipeline.py
│   │   ├── extract_frames.py
│   │   ├── build_reassemble_gt_suite.py
│   │   ├── extract_reassemble_frames.py
│   │   ├── build_multisource_sft_eval_split.py
│   │   ├── reassemble_utils.py
│   │   ├── render_reassemble_pilot_cards.py
│   │   ├── aist/
│   │   └── rh20t/
│   ├── manual_audit/
│   │   ├── gt_audit/
│   │   └── semantic_affordance_audit/
│   ├── benchmark_v1_curated.jsonl
│   ├── benchmark_v1_curated_by_type/
│   ├── benchmark_v1_frames_tbinary_20260330/
│   ├── rh20t_benchmark_v0_curated.jsonl
│   ├── rh20t_benchmark_v0_curated_by_type/
│   ├── rh20t_benchmark_v0_frames/
│   ├── reassemble_benchmark_v0_curated.jsonl
│   ├── reassemble_benchmark_v0_curated_by_type/
│   ├── reassemble_benchmark_v0_frames/
│   ├── aist_benchmark_v0/
│   │   ├── aist_benchmark_v0_curated.jsonl
│   │   ├── aist_benchmark_v0_summary.json
│   │   └── curated_by_type/
│   ├── aist_benchmark_v0_frames/
│   ├── splits_v1/
│   │   ├── gm100_sft.jsonl
│   │   ├── gm100_eval.jsonl
│   │   ├── rh20t_sft.jsonl
│   │   ├── rh20t_eval.jsonl
│   │   ├── reassemble_sft.jsonl
│   │   ├── reassemble_eval.jsonl
│   │   ├── aist_sft.jsonl
│   │   ├── aist_eval.jsonl
│   │   ├── all_sft_merged.jsonl
│   │   ├── all_eval_merged.jsonl
│   │   ├── split_summary.json
│   │   └── eval_manifest.json
│   └── eval_results_v1/
│       └── splits_v1/
└── README.md
```

## 3. Core Artifacts

### 3.1 Curated Benchmark Inputs

| Source | JSONL | Frame Cache |
| --- | --- | --- |
| GM-100 | `benchmark/benchmark_v1_curated.jsonl` | `benchmark/benchmark_v1_frames_tbinary_20260330/` |
| RH20T | `benchmark/rh20t_benchmark_v0_curated.jsonl` | `benchmark/rh20t_benchmark_v0_frames/` |
| REASSEMBLE | `benchmark/reassemble_benchmark_v0_curated.jsonl` | `benchmark/reassemble_benchmark_v0_frames/` |
| AIST | `benchmark/aist_benchmark_v0/aist_benchmark_v0_curated.jsonl` | `benchmark/aist_benchmark_v0_frames/` |

### 3.2 Split Artifacts

| File | Meaning |
| --- | --- |
| `benchmark/splits_v1/gm100_sft.jsonl` | GM-100 SFT split |
| `benchmark/splits_v1/gm100_eval.jsonl` | GM-100 Eval split |
| `benchmark/splits_v1/rh20t_sft.jsonl` | RH20T SFT split |
| `benchmark/splits_v1/rh20t_eval.jsonl` | RH20T Eval split |
| `benchmark/splits_v1/reassemble_sft.jsonl` | REASSEMBLE SFT split |
| `benchmark/splits_v1/reassemble_eval.jsonl` | REASSEMBLE Eval split |
| `benchmark/splits_v1/aist_sft.jsonl` | AIST SFT split |
| `benchmark/splits_v1/aist_eval.jsonl` | AIST Eval split |
| `benchmark/splits_v1/all_sft_merged.jsonl` | 四源合并 SFT 集 |
| `benchmark/splits_v1/all_eval_merged.jsonl` | 四源合并 Eval 集 |
| `benchmark/splits_v1/split_summary.json` | split 统计摘要 |
| `benchmark/splits_v1/eval_manifest.json` | one-click eval manifest |

## 4. Data Flow

### 4.1 Per-source GT Build

```text
raw dataset
    |
    v
dataset-specific GT builders
    |
    v
curated jsonl
    |
    v
frame extraction
    |
    v
frame cache
```

对应关系：

- GM-100：`gt_build/build_sampling_pipeline.py` -> `benchmark_v1_curated.jsonl`
- RH20T：`benchmark/gt_build/rh20t/*` -> `rh20t_benchmark_v0_curated.jsonl`
- REASSEMBLE：`gt_build/build_reassemble_gt_suite.py` -> `reassemble_benchmark_v0_curated.jsonl`
- AIST：`benchmark/gt_build/aist/*` -> `aist_benchmark_v0/aist_benchmark_v0_curated.jsonl`

### 4.2 Split Build

```text
four curated jsonl files
    |
    v
gt_build/build_multisource_sft_eval_split.py
    |
    v
benchmark/splits_v1/
  - per-source sft/eval jsonl
  - merged sft/eval jsonl
  - split_summary.json
  - eval_manifest.json
```

split 规则：

- `episode-level` 隔离
- 同一 `episode / recording / scene` 只属于一个 split
- `REASSEMBLE T10/T11/T12` 在 eval 选择时加权提高

### 4.3 Unified Eval

```text
benchmark/splits_v1/eval_manifest.json
    |
    v
eval_v1/run_all_eval_sets.py
    |
    +--> run_pilot_eval.py on GM-100 eval split
    +--> run_pilot_eval.py on RH20T eval split
    +--> run_pilot_eval.py on REASSEMBLE eval split
    +--> run_pilot_eval.py on AIST eval split
    |
    v
benchmark/eval_results_v1/splits_v1/*.jsonl
    |
    v
eval_v1/score_all_eval_sets.py
    |
    v
score_all_eval_sets_summary.json
```

设计原因：

- 四个数据集分属四个 frame cache
- 当前统一执行器 `run_pilot_eval.py` 仍以单套 frame-dir 参数工作
- 最稳的主线是 manifest + 顺序调度四个 dataset eval

## 5. Key Script Index

| Script | Role | Main Output |
| --- | --- | --- |
| `benchmark/gt_build/build_multisource_sft_eval_split.py` | 构建四源 `85/15` SFT/Eval split | `benchmark/splits_v1/*` |
| `benchmark/eval_v1/run_all_eval_sets.py` | 一键运行四个 eval split | `benchmark/eval_results_v1/splits_v1/*_eval_results.jsonl` |
| `benchmark/eval_v1/score_all_eval_sets.py` | 汇总四个 eval split 结果 | `score_all_eval_sets_summary.json` |
| `benchmark/eval_v1/run_pilot_eval.py` | 通用推理执行器 | per-item result jsonl |
| `benchmark/eval_v1/score_pilot.py` | 通用评分器 | console summary |

## 6. Current Counts

### 6.1 Full Benchmark

| Source | Items |
| --- | ---: |
| GM-100 | `15,500` |
| RH20T | `15,800` |
| REASSEMBLE | `17,165` |
| AIST | `9,427` |
| Total | `57,892` |

### 6.2 Eval Split

| Source | Eval Items | Eval Groups |
| --- | ---: | ---: |
| GM-100 | `2,643` | `628` |
| RH20T | `2,422` | `89` |
| REASSEMBLE | `2,562` | `6` |
| AIST | `1,424` | `30` |
| Total | `9,051` | - |

### 6.3 SFT Split

| Source | SFT Items |
| --- | ---: |
| GM-100 | `12,857` |
| RH20T | `13,378` |
| REASSEMBLE | `14,603` |
| AIST | `8,003` |
| Total | `48,841` |

## 7. Current Conventions

1. 论文命名使用 `T5/T8/T9`，工程兼容 `T_progress/T_temporal/T_binary`。
2. AIST 当前包含 `T3/T4/T6/T8/T9`。其中工程文件中 `T8` 落成 `T_temporal`，`T9` 保持两帧先后判别。
3. `run_pilot_eval.py` 已兼容 AIST `T9` 双帧读取。
4. `REASSEMBLE` 默认视角固定为 `hand`。
5. `RH20T` 默认视角固定为当前选定主相机。
6. `splits_v1` 是当前默认 SFT/Eval protocol。

## 8. Default Entrypoints

### 8.1 Rebuild Split

```bash
cd /data/projects/GM-100

python3 benchmark/gt_build/build_multisource_sft_eval_split.py
```

### 8.2 Run All Eval Sets

```bash
cd /data/projects/GM-100

python3 benchmark/eval_v1/run_all_eval_sets.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --api-base http://35.220.164.252:3888/v1/ \
  --api-key YOUR_KEY
```

### 8.3 Score All Eval Sets

```bash
cd /data/projects/GM-100

python3 benchmark/eval_v1/score_all_eval_sets.py
```

## 9. Document Responsibilities

1. `README.md`：只保留最高频命令。
2. `ARCHITECTURE_MAP.md`：记录当前稳定目录、数据流、默认入口。
3. `NEURIPS_ED_MASTER_PLAN.md`：记录论文主线、数据源角色、当前闭环与风险边界。
4. `benchmark_card.md`：对外叙述 benchmark 定义、可靠性与 caveats。
