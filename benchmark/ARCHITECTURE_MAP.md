# GM-100 Benchmark Architecture Map

更新时间：2026-03-31  
适用范围：`/data/projects/GM-100/benchmark`

## 1) 目录架构（按职责，重构后）

```text
GM-100/
├── gm100-cobotmagic-lerobot/                           # 原始数据集（task_00001~00110）
│   └── task_xxxxx/
│       ├── data/chunk-000/episode_XXXXXX.parquet
│       ├── videos/chunk-000/observation.images.camera_*/episode_XXXXXX.mp4
│       └── meta/{info.json,episodes.jsonl,tasks.jsonl}
├── GM100_bimanual_fullscan_20260318/                  # task 元数据与分型
│   └── task_type_annotation.csv
├── GM100_eda_plots_20260318/                          # 早期 EDA 可视化
└── benchmark/
    ├── eval_v1/                                       # 评测执行与评分入口（当前主入口）
    │   ├── run_pilot_eval.py                          # 通用推理执行器（被 v1 wrapper 复用）
    │   ├── run_benchmark_v1_eval.py                   # benchmark_v1 评测入口
    │   ├── score_pilot.py                             # 通用评分器
    │   └── score_benchmark_v1.py                      # benchmark_v1 评分入口（T6 baseline=0.5）
    ├── gt_build/                                      # GT 构建、采样、抽帧主链路
    │   ├── segmentation.py
    │   ├── build_t1_gt.py / build_t2_gt.py / ...
    │   ├── build_sampling_pipeline.py
    │   └── extract_frames.py
    ├── legacy_collision_vqa/                          # 早期 collision->grid->VQA 探索链路
    ├── manual_checks_20260325/                        # 误差分析产物
    ├── previous_results/                              # 历史 run 和旧产物归档
    │   └── manual_checks_20260320/                    # full GT 历史产物（已归档）
    ├── benchmark_v1_curated.jsonl                     # 根目录当前 latest 8 题型入口（15500）
    ├── benchmark_v1_curated_healthcheck.json          # 根目录 latest 版健康检查摘要
    ├── benchmark_v1_frames_tbinary_20260330/          # 当前 root latest 抽帧缓存
    ├── eval_results_v1/                               # 模型推理结果 JSONL
    ├── NEXT_STEPS.md
    └── ARCHITECTURE_MAP.md
```

## 2) 数据流（主线）

### 2.1 benchmark_v1 主流程（当前主线）

```text
task_type_annotation.csv
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
        |
        v
benchmark_v1_curated.jsonl  (+ healthcheck json)
        |
        v
gt_build/extract_frames.py
        |
        v
benchmark_v1_frames*/ *.jpg
        |
        v
eval_v1/run_benchmark_v1_eval.py (复用 eval_v1/run_pilot_eval.py)
        |
        v
eval_results_v1/*.jsonl
        |
        v
eval_v1/score_benchmark_v1.py
        |
        v
按 task_type / arm_type 的 acc 与 baseline 对比
```

### 2.2 历史支线（legacy，仅回溯时使用）

```text
legacy_collision_vqa/detect_collision_multitask.py
 -> export_collision_clips.py
 -> build_collision_event_grid_images.py
 -> align_collision_grids_with_parquet.py
 -> generate_vqa_from_collision_grids.py / generate_vqa_from_aligned_events.py
 -> render_vqa_cards.py
```

说明：`legacy_collision_vqa/` 用于早期探索与可视化，不是 benchmark_v1 默认评测链路。

## 3) 脚本输入输出索引（核心）

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
| `gt_build/build_sampling_pipeline.py` | 按配额+task cap 采样合并 8 题型（current root latest） | 各 `*_gt_items.jsonl` + dataset 视频可用性 | `benchmark_v1_curated.jsonl`, healthcheck json |
| `gt_build/extract_frames.py` | 按 QA 条目批量抽 JPEG（含多帧后缀） | `benchmark_v1_curated.jsonl` + mp4 | `benchmark_v1_frames*/**.jpg`, `extract_summary.json` |
| `eval_v1/run_pilot_eval.py` | 通用推理执行器（prompt/parse/retry/preflight） | input jsonl + frame dir + api/model/key | 结果 JSONL + 汇总统计 |
| `eval_v1/run_benchmark_v1_eval.py` | benchmark_v1 默认参数封装入口 | 默认指向 v1 curated 与 v1 frame dir | `eval_results_v1/*.jsonl` |
| `eval_v1/score_pilot.py` | 通用评分器（含 baseline 对比） | eval result jsonl | 控制台分数摘要 |
| `eval_v1/score_benchmark_v1.py` | benchmark_v1 评分入口（T6 baseline=0.5） | `eval_results_v1/*.jsonl` | 控制台分数摘要 |

## 4) benchmark_v1 关键约定

1. 根目录当前 latest 口径：`T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`，总量 `15500`。  
2. 当前 root release 的源快照位于：`previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/benchmark_v1_curated.jsonl`。  
3. `T_binary` 当前 root 口径使用 `composite_panel_v2`：两帧先渲染为单张 comparison image，答案标签为 `X/Y`；GT 仍仅由 `frame_index` 比较决定。  
4. T6 当前 root 口径为二分类：`actively_moving` vs `stationary`，随机基线 `0.5`。  
5. 历史 7 题型 root 快照已归档到：`previous_results/root_release_snapshots/benchmark_v1_root_7family_20260331_before_latest_promotion/`。  
6. 评分默认忽略 `INVALID/ERROR/MISSING_FRAME`（`eval_v1/score_pilot.py` 默认行为）；评测入口优先用 `eval_v1/run_benchmark_v1_eval.py` 与 `eval_v1/score_benchmark_v1.py`。  

## 5) 可直接复制的运行命令（重构后）

### 5.1 最短链路（你当前主要入口）

```bash
cd /data/projects/GM-100/benchmark

# 推荐提前设置环境变量（OpenAI/DashScope/Anthropic 任一）
# export OPENAI_API_KEY=xxx
# export DASHSCOPE_API_KEY=xxx

python eval_v1/run_benchmark_v1_eval.py \
  --model qwen3-vl-plus

python eval_v1/score_benchmark_v1.py \
  eval_results_v1/benchmark_v1_qwen3vl_plus_full.jsonl
```

说明：`score_benchmark_v1.py` 的默认输入文件名与 `run_benchmark_v1_eval.py` 默认输出文件名不完全一致，建议显式传入结果路径（如上）。

### 5.2 从采样到评分的一键全链路（推荐脚本入口）

推荐直接使用：`benchmark/run_v1_pipeline.sh`。该脚本已封装采样、抽帧、评测与评分，避免文档命令与代码逻辑漂移。

```bash
cd /data/projects/GM-100/benchmark
./run_v1_pipeline.sh --model qwen3-vl-plus
```

常用参数示例：

```bash
# 固定实验名
./run_v1_pipeline.sh --run-tag exp_001 --model qwen3-vl-plus

# 只评分已有结果
./run_v1_pipeline.sh --skip-sampling --skip-extract --skip-eval \
  --result-jsonl eval_results_v1/your_result.jsonl

# 跳过采样和抽帧，直接用已有 curated + frame 继续评测
./run_v1_pipeline.sh --skip-sampling --skip-extract \
  --result-jsonl eval_results_v1/benchmark_v1_resume.jsonl
```

查看完整参数：

```bash
./run_v1_pipeline.sh --help
```

### 5.3 当前执行口径补充（2026-03-26）

1. 根目录 `benchmark_v1_curated.jsonl` 与 `benchmark_v1_curated_healthcheck.json` 现已是 current root latest release，可直接闭环：
   - `curated -> 抽帧 -> 评测 -> 评分`。
2. 当前 root latest release 对应的推荐 frame cache 为：`benchmark_v1_frames_tbinary_20260330`。
3. 若要跑“采样->抽帧->评测->评分”全链路，默认 `GT_DIR` 已指向 `previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2`。
4. 只有在复现实验历史快照时，才需要显式改 `--gt-dir` / `--input` / `--frame-dir-default`。

### 5.4 最大链路闭环评测（当前结构可直接执行）

```bash
cd /data/projects/GM-100/benchmark
./run_v1_pipeline.sh \
  --gt-dir previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2 \
  --model qwen3-vl-plus
```

说明：该命令覆盖“采样 -> 抽帧 -> 评测 -> 评分”全链路；若只做评测可改为 `--skip-sampling --skip-extract` 并显式传 `--result-jsonl`。

## 6) 路径与可移植性约定（GitHub 友好）

1. 脚本默认路径统一用“基于 `__file__` 的仓库相对路径”，避免写死 `/data/projects/...`。  
2. API key 不落盘、不硬编码；仅用 `--api-key` 或环境变量。  
3. 大体量产物目录（如 `benchmark_v1_frames*`, `eval_results_v1/*.jsonl`）默认不进仓库。  
4. 每次重构目录后，先更新本文件，再更新 `NEXT_STEPS.md` 的默认入口说明。  

## 7) 后续维护建议

1. 增加 `benchmark/README.md`，只保留“3 条最常用命令”（采样、评测、评分）。  
2. 增加 `benchmark/configs/`（例如 `benchmark_v1.yaml`）集中管理 model/api/path 参数。  
3. 提供 `benchmark/run_v1_pipeline.sh`，把 5.2 的命令固化成单脚本入口。  
4. 在 CI 中至少加 `python -m py_compile benchmark/eval_v1/*.py benchmark/gt_build/*.py` 做语法守门。  
