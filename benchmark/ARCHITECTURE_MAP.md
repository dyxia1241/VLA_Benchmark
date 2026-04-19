# Benchmark Architecture Map

更新时间：2026-04-14  
适用范围：`/data/projects/GM-100/benchmark`

定位说明：

- 本文件记录当前主线的稳定目录结构、数据流和默认入口。
- 当前范围同时覆盖 `GM-100` root benchmark 和并行的 `REASSEMBLE benchmark-v0` side release。
- `NEURIPS_ED_MASTER_PLAN.md` 负责记录论文主线、近期决策、风险判断和滚动 next actions。
- `README.md` 只保留最常用运行命令。

## 1) 目录架构（按当前主线）

```text
GM-100/
├── gm100-cobotmagic-lerobot/                           # 原始数据集（task_00001~00110）
│   └── task_xxxxx/
│       ├── data/chunk-000/episode_XXXXXX.parquet
│       ├── videos/chunk-000/observation.images.camera_*/episode_XXXXXX.mp4
│       └── meta/{info.json,episodes.jsonl,tasks.jsonl}
├── reassemble-tuwien-researchdata/                    # REASSEMBLE 官方目录（当前冻结到 test_split1）
│   ├── data/*.h5                                      # recording 级原始文件
│   ├── poses/*_poses.json
│   ├── splits/
│   └── test_split1.txt
├── GM100_bimanual_fullscan_20260318/                  # 早期 fullscan / 分型 provenance
├── GM100_eda_plots_20260318/                          # 早期 EDA 可视化
└── benchmark/
    ├── GM100 List.xlsx                                # task 描述 + object inventory 工作簿
    ├── README.md                                      # 最短操作说明
    ├── benchmark_card.md                              # benchmark 对外描述与验证说明
    ├── run_v1_pipeline.sh                             # 采样 -> 抽帧 -> 评测 -> 评分一键入口
    ├── eval_v1/                                       # 当前评测执行与评分主入口
    │   ├── run_pilot_eval.py                          # 通用推理执行器（当前兼容 GM-100 + REASSEMBLE）
    │   ├── run_benchmark_v1_eval.py                   # benchmark_v1 默认 wrapper
    │   ├── run_reassemble_benchmark_v0_eval.py        # REASSEMBLE benchmark-v0 默认 wrapper
    │   ├── score_pilot.py                             # 通用评分器
    │   ├── score_benchmark_v1.py                      # benchmark_v1 评分入口（T6 baseline=0.5）
    │   └── score_reassemble_benchmark_v0.py           # REASSEMBLE benchmark-v0 评分入口
    ├── gt_build/                                      # GT 构建、采样、抽帧主链路
    │   ├── task_type_annotation.csv                   # 当前主线 task 元数据
    │   ├── segmentation.py
    │   ├── build_t1_gt.py / build_t2_gt.py / ...
    │   ├── build_sampling_pipeline.py
    │   ├── extract_frames.py
    │   ├── reassemble_utils.py
    │   ├── build_reassemble_gt_suite.py
    │   ├── extract_reassemble_frames.py
    │   ├── sample_reassemble_pilot.py
    │   └── render_reassemble_pilot_cards.py
    ├── manual_audit/                                  # 人工审计与 task-level semantic/affordance workflow
    │   ├── gt_audit/
    │   │   ├── audit_guideline_v1.md
    │   │   ├── export_audit_subset.py
    │   │   ├── score_audit_annotations.py
    │   │   ├── full_audit_v1/
    │   │   └── t_progress_v2_pilot_20260414/
    │   └── semantic_affordance_audit/
    │       ├── README.md
    │       ├── build_annotation_tables_v2.py
    │       ├── build_cluster_alignment_tables_v2.py
    │       ├── build_reassemble_index_and_prefill_v1.py
    │       ├── build_reassemble_recording_sequence_v1.py
    │       ├── specs/
    │       ├── catalogs/
    │       ├── derived/
    │       └── tables/
    ├── legacy_collision_vqa/                          # 早期 collision->grid->VQA 探索链路
    ├── manual_checks_20260325/                        # 历史误差分析 sidecar 产物
    ├── previous_results/                              # 历史 run、旧快照与归档
    │   ├── manual_checks_20260320/
    │   │   └── root_release_source_20260414_tprogress_v2/
    │   └── root_release_snapshots/
    ├── benchmark_v1_curated.jsonl                     # 根目录当前 latest 8 题型入口（15500）
    ├── benchmark_v1_curated_by_type/                  # 按题型拆分的 root latest 产物
    ├── benchmark_v1_curated_healthcheck.json          # root latest 健康检查摘要
    ├── benchmark_v1_frames_tbinary_20260330/          # root latest 默认 frame cache
    ├── reassemble_test_split1_suite_v0/               # REASSEMBLE 全量 suite GT
    ├── reassemble_test_split1_pilot_v0/               # REASSEMBLE reviewer-facing pilot 包
    ├── reassemble_benchmark_v0_curated.jsonl          # REASSEMBLE 当前冻结 benchmark-v0 入口（17165）
    ├── reassemble_benchmark_v0_curated_by_type/       # REASSEMBLE 按题型拆分产物
    ├── reassemble_benchmark_v0_frames/                # REASSEMBLE hand-view frame cache
    ├── reassemble_benchmark_v0_summary.json           # REASSEMBLE 冻结版本摘要
    ├── eval_results_v1/                               # 模型结果 JSONL 输出目录
    ├── NEURIPS_ED_MASTER_PLAN.md                      # 论文主线、风险边界与路线图
    └── ARCHITECTURE_MAP.md
```

## 2) 数据流（当前主线）

### 2.1 benchmark_v1 主评测链

```text
benchmark/gt_build/task_type_annotation.csv
        +
gm100-cobotmagic-lerobot(task parquet/mp4)
        |
        v
gt_build/build_t1,t2,t3,t4,t6,t_temporal,t_binary,t_progress_gt.py
        |
        v
source_snapshot/*_gt_items.jsonl
        |
        v
gt_build/build_sampling_pipeline.py
  - 按配额采样
  - 按 task cap 控制覆盖
  - 对 T3/T4/T6/T_temporal/T_binary 做 episode head/tail 90-frame boundary filter
        |
        v
benchmark_v1_curated.jsonl
benchmark_v1_curated_by_type/
benchmark_v1_curated_healthcheck.json
        |
        v
gt_build/extract_frames.py
        |
        v
benchmark_v1_frames*/ *.jpg
        |
        v
eval_v1/run_benchmark_v1_eval.py
  (wrapper over run_pilot_eval.py)
        |
        v
eval_results_v1/*.jsonl
        |
        v
eval_v1/score_benchmark_v1.py
        |
        v
按 task_type / arm_type / baseline 输出结果
```

### 2.2 manual audit 主链

```text
benchmark_v1_curated.jsonl
        +
benchmark_v1_frames_tbinary_20260330/
        |
        v
manual_audit/gt_audit/export_audit_subset.py
        |
        v
manual_audit/gt_audit/full_audit_v1/ or manual_audit/gt_audit/<your_audit_dir>/
  - audit_subset.jsonl
  - audit_items.csv
  - annotator_*.csv
  - audit_cards/*.jpg
        |
        v
annotator A / B 独立标注
        |
        v
manual_audit/gt_audit/score_audit_annotations.py
        |
        v
agreement / gt_correctness / error category 汇总
```

说明：

- `full_audit_v1/` 是当前通用正式人工审计工作目录（非 `T_progress v2`）。
- `t_progress_v2_pilot_20260414/` 是 `T_progress v2` 的定向试标包。
- 如需新的试标包，应通过 `export_audit_subset.py --output-dir ...` 重新导出。

### 2.3 semantic_affordance_audit 主链

```text
manual_audit/semantic_affordance_audit/catalogs/
  - gm100_task_catalog_v1.json
  - rh20t_task_catalog_v1.json
  - reassemble_task_catalog_v1.json
        |
        v
manual_audit/semantic_affordance_audit/build_annotation_tables_v2.py
        |
        v
tables/gm100_canonical_chain_{master,huiting_ji}_v2.{csv,xlsx}
tables/rh20t_canonical_chain_{master,huiting_ji}_v2.{csv,xlsx}
tables/reassemble_canonical_chain_{master,huiting_ji}_v2.{csv,xlsx}

benchmark/primitive_cluster/runs/one_episode_per_task_v0/cluster_proposals.jsonl
        +
canonical chain master tables
        |
        v
manual_audit/semantic_affordance_audit/build_cluster_alignment_tables_v2.py
        |
        v
tables/gm100_cluster_alignment_{master,huiting_ji}_v2.{csv,xlsx}
tables/rh20t_cluster_alignment_{master,huiting_ji}_v2.{csv,xlsx}
tables/reassemble_cluster_alignment_{master,huiting_ji}_v2.{csv,xlsx}

reassemble-tuwien-researchdata/data/*.h5
        +
reassemble-tuwien-researchdata/poses/*_poses.json
        |
        v
manual_audit/semantic_affordance_audit/build_reassemble_index_and_prefill_v1.py
        |
        v
derived/reassemble_recording_index_v1.jsonl
derived/reassemble_segment_index_v1.csv
derived/reassemble_action_prototypes_v1.csv
derived/reassemble_vocab_v1.json

derived/reassemble_recording_index_v1.jsonl
        +
derived/reassemble_segment_index_v1.csv
        |
        v
manual_audit/semantic_affordance_audit/build_reassemble_recording_sequence_v1.py
        |
        v
tables/reassemble_recording_sequence_v1.{csv,xlsx}
```

说明：

- 当前人工标注主线已经收敛到 `primitive + object` 两层，不再维护旧版 affordance-heavy 模板。
- `GM-100 / RH20T / REASSEMBLE` 都使用 canonical chain 主表；cluster alignment 仅作为 episode-level 对齐 sidecar。
- 这条链不直接参与 benchmark 默认打分，但服务于 shared primitive 叙述、T5/T10/T11/T12 设计和误差分析。

### 2.4 REASSEMBLE benchmark-v0 主链

```text
reassemble-tuwien-researchdata/data/*.h5
        +
reassemble-tuwien-researchdata/test_split1.txt
        |
        v
gt_build/build_reassemble_gt_suite.py
  - high-level segment -> T1/T2/T_progress/T6/T7/T_temporal/T_binary
  - low-level segment -> T10/T11/T12
  - default camera: hand
  - no T3/T4
        |
        v
reassemble_test_split1_suite_v0/*.jsonl
reassemble_test_split1_suite_v0/summary.json
        |
        v
current frozen release materialization
        |
        v
reassemble_benchmark_v0_curated.jsonl
reassemble_benchmark_v0_curated_by_type/
reassemble_benchmark_v0_summary.json
        |
        v
gt_build/extract_reassemble_frames.py
        |
        v
reassemble_benchmark_v0_frames/*.jpg
        |
        +--> gt_build/sample_reassemble_pilot.py
        |        |
        |        v
        |   reassemble_test_split1_pilot_v0/sample.jsonl
        |        |
        |        v
        |   gt_build/render_reassemble_pilot_cards.py
        |        |
        |        v
        |   reassemble_test_split1_pilot_v0/pilot_cards/*.jpg
        |
        v
eval_v1/run_reassemble_benchmark_v0_eval.py
        |
        v
eval_results_v1/reassemble_benchmark_v0_*.jsonl
        |
        v
eval_v1/score_reassemble_benchmark_v0.py
```

说明：

- 当前冻结源为官方 `test_split1` 的 `37` 个 `.h5 recording`。
- 默认监督与评测视角固定为 `hand`；`hama1/hama2` 只保留给 camera-shift OOD analysis。
- 当前 benchmark-v0 支持 `T1/T2/T_progress/T6/T7/T_temporal/T_binary/T10/T11/T12`，不做 `T3/T4`。

### 2.5 历史支线（legacy，仅回溯时使用）

```text
legacy_collision_vqa/detect_collision_multitask.py
 -> export_collision_clips.py
 -> build_collision_event_grid_images.py
 -> align_collision_grids_with_parquet.py
 -> generate_vqa_from_collision_grids.py / generate_vqa_from_aligned_events.py
 -> render_vqa_cards.py
```

说明：`legacy_collision_vqa/` 是早期探索链路，不是当前 benchmark_v1、manual audit 或 task taxonomy 的默认入口。

## 3) 核心脚本输入输出索引

| 脚本 | 功能 | 主要输入 | 主要输出 |
|---|---|---|---|
| `gt_build/segmentation.py` | 接触事件检测、轨迹阶段分段公共逻辑 | parquet 时序 + task meta | 阶段标签/事件（内存对象） |
| `gt_build/build_t1_gt.py` | T1 阶段分类题 GT | dataset + annotation csv | `t1_gt_items.jsonl` |
| `gt_build/build_t2_gt.py` | T2 接触/稳定性相关 GT | dataset + annotation csv | `t2_gt_items.jsonl` |
| `gt_build/build_t3_gt.py` | T3 主运动方向 GT（视觉方向映射） | dataset + annotation csv | `t3_gt_items.jsonl` |
| `gt_build/build_t4_bimanual_gt.py` | T4 双臂状态 GT | dataset + annotation csv | `t4_gt_items.jsonl` |
| `gt_build/build_t6_gt.py` | T6 二分类（actively moving / stationary）GT | dataset + annotation csv | `t6_gt_items.jsonl`, `t6_gt_summary.json` |
| `gt_build/build_t_temporal_gt.py` | T_temporal 三帧时序排序 GT | dataset + annotation csv | `t_temporal_gt_items.jsonl` |
| `gt_build/build_t_binary_gt.py` | T_binary 两帧先后判别 GT | dataset + annotation csv | `t_binary_gt_items.jsonl` |
| `gt_build/build_t_progress_gt.py` | T_progress 局部步骤进度 GT（3 档） | dataset + annotation csv | `t_progress_gt_items.jsonl` |
| `gt_build/build_sampling_pipeline.py` | 合并采样 8 题型，并应用 boundary filter | 各 `*_gt_items.jsonl` + dataset 视频可用性 | `benchmark_v1_curated.jsonl`, by-type jsonl, healthcheck json |
| `gt_build/extract_frames.py` | 批量抽取单帧/多帧 JPEG | `benchmark_v1_curated.jsonl` + mp4 | `benchmark_v1_frames*/**.jpg`, `extract_summary.json` |
| `manual_audit/gt_audit/export_audit_subset.py` | 从 curated benchmark 分层抽样并导出审计包 | curated jsonl + frame dir + quota config | audit csv/jsonl, `audit_cards/*.jpg`, `audit_summary.json` |
| `manual_audit/gt_audit/score_audit_annotations.py` | 合并 annotator CSV 并统计 agreement / error 分布 | annotator A/B csv | 审计统计摘要、冲突项汇总 |
| `manual_audit/semantic_affordance_audit/build_annotation_tables_v2.py` | 生成三数据集 canonical primitive chain 主表 | dataset catalogs + shared primitive spec | canonical-chain master / annotator 表 |
| `manual_audit/semantic_affordance_audit/build_cluster_alignment_tables_v2.py` | 生成 cluster -> canonical step 对齐表 | canonical-chain 主表 + cluster proposals | cluster-alignment master / annotator 表 |
| `manual_audit/semantic_affordance_audit/build_reassemble_index_and_prefill_v1.py` | 提取 REASSEMBLE action/segment 索引并预填 canonical chain | REASSEMBLE `.h5` + poses | `derived/reassemble_*` + REASSEMBLE canonical-chain 预填表 |
| `manual_audit/semantic_affordance_audit/build_reassemble_recording_sequence_v1.py` | 导出 recording 级 high/low-level action 序列表 | `derived/reassemble_recording_index_v1.jsonl`, `derived/reassemble_segment_index_v1.csv` | `tables/reassemble_recording_sequence_v1.{csv,xlsx}` |
| `gt_build/reassemble_utils.py` | REASSEMBLE `.h5` / segment / camera 访问公共函数 | dataset root + split/index files | recording-level helper objects（内存对象） |
| `gt_build/build_reassemble_gt_suite.py` | 构建 REASSEMBLE benchmark-v0 全量 GT suite | `reassemble-tuwien-researchdata/data/*.h5` + split txt | `reassemble_test_split1_suite_v0/*.jsonl`, `summary.json` |
| `gt_build/extract_reassemble_frames.py` | 批量抽取 REASSEMBLE benchmark-v0 所需 JPEG | REASSEMBLE curated jsonl + `.h5` | `reassemble_benchmark_v0_frames/*.jpg`, `extract_summary.json` |
| `gt_build/sample_reassemble_pilot.py` | 从 REASSEMBLE 全量题库抽 pilot | full curated jsonl + quotas | `reassemble_test_split1_pilot_v0/*.jsonl`, summary |
| `gt_build/render_reassemble_pilot_cards.py` | 渲染 REASSEMBLE pilot 卡片 | pilot jsonl + frame dir | `reassemble_test_split1_pilot_v0/pilot_cards/*.jpg` |
| `eval_v1/run_pilot_eval.py` | 通用推理执行器，支持 parse/retry/preflight、可选 task-meta prepend，当前兼容 GM-100 + REASSEMBLE item schema | input jsonl + frame dir + api/model/key + task-meta xlsx | 结果 JSONL + 汇总统计 |
| `eval_v1/run_benchmark_v1_eval.py` | benchmark_v1 默认评测 wrapper | 默认指向 root latest curated 与 frame dir | `eval_results_v1/*.jsonl` |
| `eval_v1/run_reassemble_benchmark_v0_eval.py` | REASSEMBLE benchmark-v0 默认评测 wrapper | 默认指向 REASSEMBLE curated 与 frame dir | `eval_results_v1/reassemble_benchmark_v0_*.jsonl` |
| `eval_v1/score_pilot.py` | 通用评分器（当前兼容 GM-100 + REASSEMBLE） | eval result jsonl | 控制台分数摘要 |
| `eval_v1/score_benchmark_v1.py` | benchmark_v1 默认评分入口（T6 baseline=0.5） | `eval_results_v1/*.jsonl` | 控制台分数摘要 |
| `eval_v1/score_reassemble_benchmark_v0.py` | REASSEMBLE benchmark-v0 评分入口 | `eval_results_v1/reassemble_benchmark_v0_*.jsonl` | 控制台分数摘要 |

## 4) 当前关键约定

1. 根目录当前 latest benchmark 口径为 `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`，总量 `15500`。
2. 当前 root release 的 source snapshot 位于：`previous_results/manual_checks_20260320/root_release_source_20260414_tprogress_v2/`。
   - 其中 `T1/T2/T3/T4/T6/T_temporal/T_binary` 冻结为当前 root release 集合，`T_progress` 保留 full-pool v2 GT。
3. 当前 root latest 推荐 frame cache 为：`benchmark_v1_frames_tbinary_20260330/`。
4. `T_binary` 当前 root 口径使用 `composite_panel_v2`：两帧先渲染为单张 comparison image，答案标签为 `X/Y`；GT 仍由 `frame_index` 顺序决定。
5. T6 当前 root 口径为二分类：`actively_moving` vs `stationary`，随机基线 `0.5`。
6. `T_progress` 当前 root 口径为 `within-local-step progress`：
   - 输入为 `5` 帧有序上下文：`[-6,-3,0,+3,+6]`
   - 标签为 `A/B/C = early/middle/late`
   - 随机基线 `1/3`
7. boundary filter 当前在采样阶段实现，而不是回写到各 GT builder：
   - 过滤题型：`T3/T4/T6/T_temporal/T_binary`
   - 不过滤：`T1/T2/T_progress`
   - 默认 guard：`head_guard_frames=90`，`tail_guard_frames=90`
8. `manual_audit/gt_audit/full_audit_v1/` 是当前通用正式人工审计包：
   - 总量 `470`
   - 覆盖 `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`
   - 当前审阅入口统一使用 `audit_cards/*.jpg`；`T_binary` 所需视觉证据已直接嵌入卡片
   - 其中 `T_progress` 条目对应旧版定义；当前 `T_progress v2` 以定向试标包为准
9. `T_progress v2` 的最新定向试标包位于：`manual_audit/gt_audit/t_progress_v2_pilot_20260414/`。
10. `--prepend-task-meta` 是评测开关，不是默认 benchmark protocol：
   - 默认关闭
   - 开启后优先读取 `benchmark/GM100 List.xlsx`
   - xlsx 缺失时 fallback 到 dataset `tasks.jsonl`
11. 评分默认忽略 `INVALID/ERROR/MISSING_FRAME`；主入口优先使用 `run_benchmark_v1_eval.py` 与 `score_benchmark_v1.py`。
12. `manual_checks_20260325/` 保留为历史 targeted diagnostics，不属于当前默认执行链。
13. `REASSEMBLE benchmark-v0` 当前冻结源为官方 `test_split1` 的 `37` 个 `.h5 recording`；现阶段不再继续增量解压。
14. `REASSEMBLE` 默认监督与评测视角固定为 `hand`；`hama1/hama2` 仅保留给 OOD 分析，`capture_node-camera-image` 不进入默认 protocol。
15. `REASSEMBLE` 当前不定义 `T3/T4`；默认题库为 `T1/T2/T_progress/T6/T7/T_temporal/T_binary/T10/T11/T12`。
16. `REASSEMBLE T6` 使用 high-level segment 内相对进度多帧采样：`0.15 / 0.325 / 0.50 / 0.675 / 0.85`。
17. `REASSEMBLE T7` 使用 high-level segment 前段采样：`0.10 / 0.18 / 0.26 / 0.34`。
18. `REASSEMBLE T10/T11/T12` 保持多帧输入，并直接沿官方 low-level action vocabulary 构题，不先压到 shared primitive 闭集。

## 5) 当前默认入口

### 5.1 从 root curated 直接闭环

```bash
cd /data/projects/GM-100/benchmark

python gt_build/extract_frames.py \
  --input-jsonl benchmark_v1_curated.jsonl \
  --output-dir benchmark_v1_frames_tbinary_20260330 \
  --t6-context

python eval_v1/run_benchmark_v1_eval.py \
  --input benchmark_v1_curated.jsonl \
  --frame-dir-default benchmark_v1_frames_tbinary_20260330 \
  --model qwen3-vl-plus \
  --output eval_results_v1/benchmark_v1_qwen3vl_plus_full.jsonl

python eval_v1/score_benchmark_v1.py \
  eval_results_v1/benchmark_v1_qwen3vl_plus_full.jsonl
```

### 5.2 一键全链路

```bash
cd /data/projects/GM-100/benchmark

./run_v1_pipeline.sh \
  --gt-dir previous_results/manual_checks_20260320/root_release_source_20260414_tprogress_v2 \
  --model qwen3-vl-plus
```

### 5.3 task-meta prompt ablation

```bash
cd /data/projects/GM-100/benchmark

python eval_v1/run_benchmark_v1_eval.py \
  --input benchmark_v1_curated.jsonl \
  --frame-dir-default benchmark_v1_frames_tbinary_20260330 \
  --model qwen3-vl-plus \
  --prepend-task-meta
```

说明：开启 `--prepend-task-meta` 后的结果应单独命名、单独汇报，不能与默认 benchmark 主结果混写。

### 5.4 formal audit 评分入口

```bash
cd /data/projects/GM-100/benchmark

python manual_audit/gt_audit/score_audit_annotations.py \
  --annotator-a manual_audit/gt_audit/full_audit_v1/annotator_a.csv \
  --annotator-b path/to/annotator_b.csv
```

### 5.5 REASSEMBLE benchmark-v0 评测入口

```bash
cd /data/projects/GM-100/benchmark

python eval_v1/run_reassemble_benchmark_v0_eval.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --output eval_results_v1/reassemble_benchmark_v0_qwen25vl32b_full.jsonl

python eval_v1/score_reassemble_benchmark_v0.py \
  eval_results_v1/reassemble_benchmark_v0_qwen25vl32b_full.jsonl
```

说明：`REASSEMBLE` 结果应与 `benchmark_v1` 分开汇报；当前不参与 root latest headline score。

## 6) 文档职责约定

1. `README.md`：只保留最常用运行命令。
2. `ARCHITECTURE_MAP.md`：记录当前稳定主线的结构、数据流和入口。
3. `NEURIPS_ED_MASTER_PLAN.md`：记录论文主线、方法论风险、阶段结论与接下来的动作。
4. `benchmark_card.md`：对外叙述 benchmark 定义、验证状态和 caveats。
5. 当 root latest 口径、默认入口或主目录结构变更时，至少同步更新这四份文档中的相关项。
