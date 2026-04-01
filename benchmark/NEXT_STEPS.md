# GM-100 Benchmark Architecture Map

更新时间：2026-04-01  
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


### AF. root `benchmark_v1` 提升到 latest 口径（2026-03-31）

1. 已执行的 root release 同步（已完成）
- 根目录以下产物已提升为当前 latest 口径：
  - `benchmark/benchmark_v1_curated.jsonl`
  - `benchmark/benchmark_v1_curated_healthcheck.json`
  - `benchmark/benchmark_v1_curated_by_type/`
- 当前 root release 题型：
  - `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`
- 当前 root release 总量：
  - `15,500`
- 当前 root release 的默认 frame cache：
  - `benchmark/benchmark_v1_frames_tbinary_20260330`

2. 默认入口已同步（已完成）
- 已更新：
  - `benchmark/eval_v1/run_benchmark_v1_eval.py`
  - `benchmark/gt_build/build_sampling_pipeline.py`
  - `benchmark/run_v1_pipeline.sh`
- 当前默认 source snapshot：
  - `benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2`

3. 文档已同步（已完成）
- 已更新：
  - `benchmark/README.md`
  - `benchmark/ARCHITECTURE_MAP.md`
  - `benchmark/benchmark_card.md`
- 目标：
  - 不再把根目录口径写成“冻结 7 题型版”；
  - 根目录当前即 latest root release。

4. 历史版本保留（已完成）
- 原根目录 7 题型快照已备份到：
  - `benchmark/previous_results/root_release_snapshots/benchmark_v1_root_7family_20260331_before_latest_promotion/`

5. 需要明确的 superseding note
- 本节会覆盖 AE 中“先不把 `T_binary v2` 并入根目录冻结版 benchmark”的仓库状态判断。
- 也就是说：
  - **方法论风险判断仍然保留**（`T_binary` 仍有 left-panel bias 风险）；
  - **但仓库发布口径已经前移**，当前 root `benchmark_v1` 已包含 `T_binary v2`。
- 后续 paper / card / eval 叙述必须按这个新口径执行，不能再把根目录说成旧 7 题型版。


### AG. HAKE-guided task taxonomy 方案整理（2026-03-31）

1. 当前立场
- 不把后续 task taxonomy 写成“HAKE labels”。
- 更准确的说法是：`HAKE-guided task families`。
- 原因：当前依据是 GM-100 的任务设计哲学与 task name / representative clips 的人工归纳；不是复用一套官方发布的、逐 task 对齐的 HAKE 标签表。

2. 三条轴必须拆开
- `semantic family`：任务的核心 manipulation primitive。
- `common_vs_long_tail`：分布属性，不是语义本体。
- `coordination_pattern`：执行该任务时所需的单臂/双臂协作结构。
- 后续 card / paper / annotation 表中，这三条轴必须正交表达，不能混写。

3. `coordination_pattern` 的工作定义
- 该字段不回答“任务是什么”，只回答“任务依赖怎样的手臂协作拓扑”。
- 暂定闭集：
  - `single_arm_direct`
  - `single_arm_with_support`
  - `bimanual_symmetric`
  - `bimanual_asymmetric_stabilize_manipulate`
  - `bimanual_sequential_handoff`
- 这条轴对后续双臂能力分析有必要保留；否则难以区分模型失败源于 semantic long-tail 还是源于协作结构本身。

4. 当前认为相对稳的 primary family 候选
- `transport_sort_pack`
- `open_close_access`
- `trigger_press_activate`
- `constrained_insert_attach_detach`
- `alignment_stack_assemble`
- `material_transfer_dispense`
- `tool_mediated_manipulation`
- `deformable_handling`
- `search_select_inspect`
- `dynamic_force_interaction`（可选；若 pilot 一致性差，则降级为 tag）

5. 当前不建议直接保留为 primary family 的旧写法
- `wipe_clean_spread`
  - 混合了表面清洁、变形体整理、材料铺展，视觉/物理瓶颈不一致。
- `twist_clip_unlock`
  - 太稀疏，且与 `open_close_access` / `constrained_insert_attach_detach` / `alignment_stack_assemble` 边界重叠。
- `assist_feed_cover_social`
  - 混的是场景语义，不是 manipulation primitive。
- `hit_throw_shake_activate`
  - 把动作机制和任务结果混在一起；如果保留，应该重写为 `dynamic_force_interaction`。

6. 更适合做 secondary tag / context tag 的概念
- `assistive`
- `socially_grounded`
- `handover`
- `wearable`
- `fasten_unfasten`
- `concealment`
- 这些概念保留分析价值，但不宜直接作为 primary family。

7. 当前选择 family 的判断准则
- `primitive-first`：先按核心交互原语分，不按对象名分。
- `affordance-sensitive`：如果两个任务表面动词相近，但关键几何关系/物理约束不同，应拆开。
- `annotatable`：两个 annotator 独立标注时，大多数 task 应能稳定落在同一 primary family。
- `analytically useful`：后续必须能支持 per-family gap、common vs long-tail gap、coordination gap 分析。

8. 已识别的边界风险
- `open_close_access` 与 `trigger_press_activate` 需要专门区分，不能继续合成一个大类。
- `tool_mediated_manipulation` 与 `material_transfer_dispense` 需要按“工具是核心原语还是只是载体”区分。
- `deformable_handling` 与 `transport_sort_pack` 需要按“难点是否来自 deformable affordance”区分。
- `constrained_insert_attach_detach` 与 `alignment_stack_assemble` 需要按“狭窄接口插接/挂接” vs “平面或结构对齐”区分。

9. 标注策略
- 先不全量硬标。
- 先抽约 `20` 个 task 做 pilot annotation。
- 两位 annotator 独立标：
  - `hake_family_primary`
  - `hake_family_secondary`
  - `affordance_tags`
  - `coordination_pattern`
  - `common_vs_long_tail`
  - `droid_overlap`
- 然后计算：
  - `common_vs_long_tail` 的 Cohen's kappa
  - `hake_family_primary` 的 agreement rate
  - 如时间允许，再算 `coordination_pattern` 的 kappa
- 只有 pilot 通过后，再上全量 `106/110` task。

10. 叙述口径约束
- 文档中不要写“采用 GM-100 官方 HAKE 标签”。
- 更安全的写法是：
  - GM-100 的任务设计受到 HAKE / HOI primitive / affordance 思想启发；
  - benchmark 侧新增了一层 `HAKE-guided task family` 组织与审计。
- `common_vs_long_tail` 应明确写成 benchmark-side analytic label，而不是数据集官方现成字段。

11. 下一步具体动作
- 从 `gm100-cobotmagic-lerobot/task_*/meta/tasks.jsonl` 抽取全量 task name 清单。
- 产出 `task_semantic_annotation_v1.csv` 模板。
- 产出 annotator guideline。
- 选取 20 个 task 做 pilot。
- 根据 pilot 冲突，再最终冻结 primary family 词表。

### AH. manual audit + boundary filter 当前状态（2026-04-01）

1. root `benchmark_v1` 已按“前后 1 秒过滤”覆盖更新（已完成）
- 生效位置：`gt_build/build_sampling_pipeline.py`
- 过滤对象：`T3/T4/T6/T_temporal/T_binary`
- 不过滤：`T1/T2/T_progress`
- 实现方式：在采样阶段基于 top-view 视频长度做 head/tail guard，而不是回写各 task 的 GT builder。
- 当前默认 guard：`head_guard_frames=30`，`tail_guard_frames=30`

2. 当前边界过滤的实际影响（已完成）
- root 输出已覆盖：
  - `benchmark/benchmark_v1_curated.jsonl`
  - `benchmark/benchmark_v1_curated_healthcheck.json`
  - `benchmark/benchmark_v1_curated_by_type/`
- 被边界过滤移除的候选数：
  - `T3`: `881 / 120613`（`0.73%`）
  - `T4`: `1851 / 36590`（`5.06%`）
  - `T6`: `1434 / 26929`（`5.33%`）
  - `T_temporal`: `4461 / 21725`（`20.53%`）
  - `T_binary`: `1819 / 21151`（`8.60%`）
- 合计移除：`10446`
- 重要结论：虽然删掉了一批明显位于视频开头/结尾的不可答样本，但 8 个题型的 target 仍全部补满；root benchmark 总量仍为 `15500`。

3. root frame cache 已补齐到新口径（已完成）
- 默认 frame cache 仍使用：`benchmark/benchmark_v1_frames_tbinary_20260330`
- 已补齐：
  - 新采样后缺失的单帧 JPEG
  - `T3` 的 `t-6,t-3,t0,t+3` 多帧上下文
  - `T6` 的 `t-6,t-3,t0,t+3,t+6` 多帧上下文
- 这样做的目的：保持 `eval_v1/run_benchmark_v1_eval.py` 与 `manual_audit/export_audit_subset.py` 仍可继续使用同一个 root frame dir。

4. manual audit 工具链当前已可用（已完成）
- 标注说明（简体中文）：`benchmark/manual_audit/audit_guideline_v1.md`
- 标注模板：`benchmark/manual_audit/audit_template_v1.csv`
- 导出脚本：`benchmark/manual_audit/export_audit_subset.py`
- 评分/汇总脚本：`benchmark/manual_audit/score_audit_annotations.py`
- 当前 card 形式：每道题会把“图片/多帧 + 问题 + 选项 + benchmark GT”合成在同一张 `audit card` JPG 中；annotator 只需看 card 并把判断填回 CSV。

5. 当前 full-task pilot 已更新（已完成）
- 路径：`benchmark/manual_audit/pilot_audit_v1/`
- 当前是 full-task 版，不是只含受影响 task 的 50 条子集。
- 当前配额：
  - `T1=8`
  - `T2=10`
  - `T3=10`
  - `T4=8`
  - `T6=10`
  - `T_temporal=10`
  - `T_binary=10`
  - `T_progress=10`
- 当前总量：`76`
- 已生成：
  - `audit_summary.json`
  - `annotator_a.csv`
  - `annotator_b.csv`
  - `audit_cards/*.jpg`
  - `review_assets/`（`T_binary` composite）

6. 当前已经确认的风险（需要后续审计验证）
- “前后 1 秒过滤”只能解决一部分 FOV 问题，不能保证样本一定可答。
- 目前已观察到：即使不在视频开头/结尾，top-view 里仍可能看不到机械臂，但底层物理信号仍然存在；这会继续影响 `T3/T4/T6`，以及部分 `T_temporal/T_binary` 的可答性。
- 因此：
  - 当前 1s guard 应视为 `simple first-pass filter`
  - 不能把它当作最终的 visibility validity solution

7. 当前 full-task pilot 仍保留（作为 tool/protocol smoke test）
- `benchmark/manual_audit/pilot_audit_v1/` 仍保留，作用主要是：
  - 快速检查 audit card 展示形式；
  - 快速验证 annotator guideline / CSV 字段 / scoring 脚本是否顺畅；
  - 在正式标 420 题前先做小规模冲突对齐。

8. 受影响 task-only pilot 仍保留（可选分析包）
- 路径：`benchmark/manual_audit/pilot_audit_boundary_1s_v1/`
- 用途：只看 `T3/T4/T6/T_temporal/T_binary` 五类受影响题的快速误差分析。
- 注意：默认工作目录现在应以 `benchmark/manual_audit/pilot_audit_v1/` 为准；前者只是 sidecar analysis pack。

9. 正式版 full audit 包已生成（已完成）
- 路径：`benchmark/manual_audit/full_audit_v1/`
- 题型与配额：
  - `T1=80`
  - `T2=60`
  - `T3=60`
  - `T4=60`
  - `T6=60`
  - `T_progress=50`
  - `T_temporal=50`
- 总量：`420`
- 已生成：
  - `audit_summary.json`
  - `audit_items.csv`
  - `annotator_a.csv`
  - `annotator_b.csv`
  - `audit_subset.jsonl`
  - `audit_cards/*.jpg`
- 注意：
  - 这版 formal audit 不包含 `T_binary`，与最早规划的 7-family 正式版保持一致。
  - 这次导出时 `relaxed_task_cap=true`，说明为了凑满 `420` 条，`max_per_task_id=2` 的软限制被放松过；因此少数 `task_id` 的抽样次数会超过 2。

10. `full_audit_v1` 已做完整性核对（已完成）
- 已核对通过：
  - `audit_cards/` 实际文件数 = `420`
  - `audit_items.csv` 数据行数 = `420`
  - `annotator_a.csv` 数据行数 = `420`
  - `annotator_b.csv` 数据行数 = `420`
  - `audit_subset.jsonl` 行数 = `420`
  - `audit_summary.json.selected_total = 420`
  - `audit_items.csv` / `annotator_a.csv` / `annotator_b.csv` / `audit_subset.jsonl` 的 `audit_item_id` 与 `item_index` 顺序一致
  - 全部 `audit_card_path` 均存在
  - 全部 `frame_paths` 均存在

11. 现在最应该做的事（next）
- 以 `benchmark/manual_audit/full_audit_v1/` 作为正式人工审计工作目录，开始双人独立标注。
- 标完后用 `benchmark/manual_audit/score_audit_annotations.py` 计算：
  - overall agreement
  - per-task agreement
  - `gt_correctness` 的一致性
  - 主要 error category 分布
- 如果正式 audit 中 `visual_answerability=not_answerable` 在 `T3/T4/T6/T_temporal` 中仍显著偏高，则下一步不要只继续调 head/tail 秒数，而是要加第二层 `visibility / answerability gate`。
- 在 formal audit 结果出来前，不建议把当前 root benchmark 宣称为“visibility issue 已解决”；更准确的说法应是：
  - 已做了基于 episode 边界的 first-pass filtering；
  - formal manual audit 已启动，用于验证剩余样本的可回答性与 GT 有效性。
