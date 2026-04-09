# GM-100 Benchmark Architecture Map

更新时间：2026-04-03  
适用范围：`/data/projects/GM-100/benchmark`

定位说明：

- 本文件记录当前主线的稳定目录结构、数据流和默认入口。
- `NEXT_STEPS.md` 负责记录近期决策、风险判断和滚动 next actions。
- `README.md` 只保留最常用运行命令。

## 1) 目录架构（按当前主线）

```text
GM-100/
├── gm100-cobotmagic-lerobot/                           # 原始数据集（task_00001~00110）
│   └── task_xxxxx/
│       ├── data/chunk-000/episode_XXXXXX.parquet
│       ├── videos/chunk-000/observation.images.camera_*/episode_XXXXXX.mp4
│       └── meta/{info.json,episodes.jsonl,tasks.jsonl}
├── GM100_bimanual_fullscan_20260318/                  # 早期 fullscan / 分型 provenance
├── GM100_eda_plots_20260318/                          # 早期 EDA 可视化
└── benchmark/
    ├── GM100 List.xlsx                                # task 描述 + object inventory 工作簿
    ├── README.md                                      # 最短操作说明
    ├── benchmark_card.md                              # benchmark 对外描述与验证说明
    ├── run_v1_pipeline.sh                             # 采样 -> 抽帧 -> 评测 -> 评分一键入口
    ├── eval_v1/                                       # 当前评测执行与评分主入口
    │   ├── run_pilot_eval.py                          # 通用推理执行器（支持 task-meta prepend）
    │   ├── run_benchmark_v1_eval.py                   # benchmark_v1 默认 wrapper
    │   ├── score_pilot.py                             # 通用评分器
    │   └── score_benchmark_v1.py                      # benchmark_v1 评分入口（T6 baseline=0.5）
    ├── gt_build/                                      # GT 构建、采样、抽帧主链路
    │   ├── task_type_annotation.csv                   # 当前主线 task 元数据
    │   ├── segmentation.py
    │   ├── build_t1_gt.py / build_t2_gt.py / ...
    │   ├── build_sampling_pipeline.py
    │   └── extract_frames.py
    ├── manual_audit/                                  # 人工审计与 task-level semantic/affordance workflow
    │   ├── gt_audit/
    │   │   ├── audit_guideline_v1.md
    │   │   ├── export_audit_subset.py
    │   │   ├── score_audit_annotations.py
    │   │   └── full_audit_v1/
    │   └── semantic_affordance_audit/
    │       ├── build_task_semantic_seed.py
    │       ├── build_task_affordance_template.py
    │       ├── import_gm100_list_inventory.py
    │       ├── task_semantic_annotation_v1.csv
    │       ├── task_affordance_annotation_v1.csv
    │       └── task_object_inventory_v1.json
    ├── legacy_collision_vqa/                          # 早期 collision->grid->VQA 探索链路
    ├── manual_checks_20260325/                        # 历史误差分析 sidecar 产物
    ├── previous_results/                              # 历史 run、旧快照与归档
    │   ├── manual_checks_20260320/
    │   └── root_release_snapshots/
    ├── benchmark_v1_curated.jsonl                     # 根目录当前 latest 8 题型入口（15500）
    ├── benchmark_v1_curated_by_type/                  # 按题型拆分的 root latest 产物
    ├── benchmark_v1_curated_healthcheck.json          # root latest 健康检查摘要
    ├── benchmark_v1_frames_tbinary_20260330/          # root latest 默认 frame cache
    ├── eval_results_v1/                               # 模型结果 JSONL 输出目录
    ├── NEXT_STEPS.md
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
full_gt_*/{t1,t2,t3,t4,t6,t_temporal,t_binary,t_progress}_gt_items.jsonl
        |
        v
gt_build/build_sampling_pipeline.py
  - 按配额采样
  - 按 task cap 控制覆盖
  - 对 T3/T4/T6/T_temporal/T_binary 做 episode head/tail 1s boundary filter
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

- `full_audit_v1/` 是当前正式人工审计工作目录。
- 历史 pilot audit 目录当前不再保留在仓库中；如需新的试标包，应通过 `export_audit_subset.py --output-dir ...` 重新导出。

### 2.3 semantic_affordance_audit 主链

```text
gm100-cobotmagic-lerobot/task_*/meta/tasks.jsonl
        +
benchmark/gt_build/task_type_annotation.csv
        +
benchmark/GM100 List.xlsx
        |
        v
manual_audit/semantic_affordance_audit/build_task_semantic_seed.py
        |
        v
task_semantic_annotation_v1.csv
task_semantic_annotation_annotator_*.csv

dataset task dirs
        +
task_semantic_annotation_v1.csv
        |
        v
manual_audit/semantic_affordance_audit/build_task_affordance_template.py
        |
        v
task_affordance_annotation_v1.csv
task_affordance_annotation_annotator_*.csv

GM100 List.xlsx
        +
task_affordance_annotation_v1.csv
        |
        v
manual_audit/semantic_affordance_audit/import_gm100_list_inventory.py
        |
        v
task_object_inventory_v1.json
```

说明：

- semantic 侧当前先围绕 `primitive_1 / primitive_2 / primitive_3 / coordination_pattern / common_vs_long_tail / droid_overlap` 做 primitive-first 标注；如后续需要 family，则基于裁决后的 primitive 层再统一聚合。
- affordance 侧当前围绕 `object_affordance_tags / interface_affordance_tags / constraint_affordance_tags / role_tags`。
- 这条链当前是 task-level annotation workflow，不直接参与 benchmark_v1 默认打分，但服务于后续分层分析与 benchmark 叙述。

### 2.4 历史支线（legacy，仅回溯时使用）

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
| `gt_build/build_t_progress_gt.py` | T_progress 五档进度 GT | dataset + annotation csv | `t_progress_gt_items.jsonl` |
| `gt_build/build_sampling_pipeline.py` | 合并采样 8 题型，并应用 boundary filter | 各 `*_gt_items.jsonl` + dataset 视频可用性 | `benchmark_v1_curated.jsonl`, by-type jsonl, healthcheck json |
| `gt_build/extract_frames.py` | 批量抽取单帧/多帧 JPEG | `benchmark_v1_curated.jsonl` + mp4 | `benchmark_v1_frames*/**.jpg`, `extract_summary.json` |
| `manual_audit/gt_audit/export_audit_subset.py` | 从 curated benchmark 分层抽样并导出审计包 | curated jsonl + frame dir + quota config | audit csv/jsonl, `audit_cards/*.jpg`, `audit_summary.json` |
| `manual_audit/gt_audit/score_audit_annotations.py` | 合并 annotator CSV 并统计 agreement / error 分布 | annotator A/B csv | 审计统计摘要、冲突项汇总 |
| `manual_audit/semantic_affordance_audit/build_task_semantic_seed.py` | 生成 semantic seed 标注表 | `tasks.jsonl` + `benchmark/gt_build/task_type_annotation.csv` | semantic 主表与 annotator sidecar 表 |
| `manual_audit/semantic_affordance_audit/build_task_affordance_template.py` | 生成 affordance 标注模板 | dataset task dirs + semantic seed csv | affordance 主表与 annotator sidecar 表 |
| `manual_audit/semantic_affordance_audit/import_gm100_list_inventory.py` | 从 `GM100 List.xlsx` 导入 object inventory | `GM100 List.xlsx` + affordance csv | 回填后的 affordance csv + `task_object_inventory_v1.json` |
| `eval_v1/run_pilot_eval.py` | 通用推理执行器，支持 parse/retry/preflight 和可选 task-meta prepend | input jsonl + frame dir + api/model/key + task-meta xlsx | 结果 JSONL + 汇总统计 |
| `eval_v1/run_benchmark_v1_eval.py` | benchmark_v1 默认评测 wrapper | 默认指向 root latest curated 与 frame dir | `eval_results_v1/*.jsonl` |
| `eval_v1/score_pilot.py` | 通用评分器 | eval result jsonl | 控制台分数摘要 |
| `eval_v1/score_benchmark_v1.py` | benchmark_v1 默认评分入口（T6 baseline=0.5） | `eval_results_v1/*.jsonl` | 控制台分数摘要 |

## 4) 当前关键约定

1. 根目录当前 latest benchmark 口径为 `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`，总量 `15500`。
2. 当前 root release 的 source snapshot 位于：`previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/`。
3. 当前 root latest 推荐 frame cache 为：`benchmark_v1_frames_tbinary_20260330/`。
4. `T_binary` 当前 root 口径使用 `composite_panel_v2`：两帧先渲染为单张 comparison image，答案标签为 `X/Y`；GT 仍由 `frame_index` 顺序决定。
5. T6 当前 root 口径为二分类：`actively_moving` vs `stationary`，随机基线 `0.5`。
6. boundary filter 当前在采样阶段实现，而不是回写到各 GT builder：
   - 过滤题型：`T3/T4/T6/T_temporal/T_binary`
   - 不过滤：`T1/T2/T_progress`
   - 默认 guard：`head_guard_frames=30`，`tail_guard_frames=30`
7. `manual_audit/gt_audit/full_audit_v1/` 是当前正式人工审计包：
   - 总量 `470`
   - 覆盖 `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`
   - 当前审阅入口统一使用 `audit_cards/*.jpg`；`T_binary` 所需视觉证据已直接嵌入卡片
8. `--prepend-task-meta` 是评测开关，不是默认 benchmark protocol：
   - 默认关闭
   - 开启后优先读取 `benchmark/GM100 List.xlsx`
   - xlsx 缺失时 fallback 到 dataset `tasks.jsonl`
9. 评分默认忽略 `INVALID/ERROR/MISSING_FRAME`；主入口优先使用 `run_benchmark_v1_eval.py` 与 `score_benchmark_v1.py`。
10. `manual_checks_20260325/` 保留为历史 targeted diagnostics，不属于当前默认执行链。

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
  --gt-dir previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2 \
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

## 6) 文档职责约定

1. `README.md`：只保留最常用运行命令。
2. `ARCHITECTURE_MAP.md`：记录当前稳定主线的结构、数据流和入口。
3. `NEXT_STEPS.md`：记录滚动决策、方法论风险和接下来的动作。
4. `benchmark_card.md`：对外叙述 benchmark 定义、验证状态和 caveats。
5. 当 root latest 口径、默认入口或主目录结构变更时，至少同步更新这四份文档中的相关项。
