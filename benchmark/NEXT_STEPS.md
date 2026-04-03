# GM-100 Benchmark Next Steps

更新时间：2026-04-03  
适用范围：`/data/projects/GM-100/benchmark`

定位说明：

- `ARCHITECTURE_MAP.md`：当前稳定主线的目录结构、数据流和默认入口。
- `README.md`：最常用运行命令。
- 本文件：记录近期决策、风险判断、当前 priority 和接下来的动作。

## 1) 当前默认工作口径（简版）

1. root latest benchmark 口径：
   - `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`
   - 总量 `15500`
2. 当前默认 source snapshot：
   - `benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/`
3. 当前默认 frame cache：
   - `benchmark/benchmark_v1_frames_tbinary_20260330/`
4. 当前 formal manual audit 工作目录：
   - `benchmark/manual_audit/gt_audit/full_audit_v1/`
5. 当前 task taxonomy 工作目录：
   - `benchmark/manual_audit/semantic_affordance_audit/`
6. `--prepend-task-meta` 是可选实验开关，不是默认 benchmark protocol。

## 2) 当前最重要的事（按优先级）

1. 正式推进 `manual_audit/gt_audit/full_audit_v1/` 的双人独立标注，并在标完后跑 agreement / GT correctness / error category 汇总。
2. 如果 `T3/T4/T6/T_temporal` 的 `visual_answerability=not_answerable` 仍明显偏高，就不要只继续调 head/tail 秒数，而是进入第二层 `visibility / answerability gate` 设计。
3. 启动 semantic pilot：
   - 从当前 `106` 个 semantic seed task 中选约 `20` 个 task
   - 覆盖 common / long-tail、single-arm / bimanual 和主要 primitive 区域（如 open-access / insert-attach / transfer / tool-mediated / deformable）
4. 启动 affordance pilot 前置准备：
   - 先基于 `object_inventory_raw + task_name_raw` 生成 affordance seed candidates
   - 再选 `15-20` 个 task 做 pilot
5. 如果要做 `task-meta` prompt 实验，结果必须单独命名、单独汇报，不能替代默认 benchmark 主结果。

## 3) 文档分工约定

1. 不再把目录架构、数据流、脚本 I/O 和默认命令长期双写在本文件里。
2. 结构与入口变更时，优先更新 `ARCHITECTURE_MAP.md`。
3. 本文件优先保留：
   - 口径变更
   - 方法论风险
   - 当前状态
   - 下一步动作

## 4) 近期决策与状态记录


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


### AG. primitive-first task taxonomy 方案收敛（2026-03-31，2026-04-02 更新）

1. 当前立场
- 不把当前 task taxonomy 写成“GM-100 官方 HAKE labels”。
- 也不把当前 semantic pilot 设计成 annotator 直接判 `family`。
- 更准确的说法是：当前采用 `primitive-first` 的 benchmark-side task taxonomy；如果后续需要 family，只从裁决后的 primitive 层往上统一聚合。
- 原因：当前依据是 GM-100 的任务设计哲学、task name、arm metadata 和后续 pilot 观察；不是复用一套官方发布的逐 task HAKE 标签表。

2. 当前必须拆开的三条轴
- `semantic primitive layer`：任务在测试什么 manipulation primitive，当前按 `primitive_1 / primitive_2 / primitive_3` 表达。
- `common_vs_long_tail`：分布属性，不是语义本体。
- `coordination_pattern`：执行该任务所需的单臂/双臂协作结构。
- 后续 card / paper / annotation 表中，这三条轴必须正交表达，不能混写。

3. `coordination_pattern` 的工作定义
- 该字段不回答“任务是什么”，只回答“任务依赖怎样的手臂协作拓扑”。
- 暂定闭集：
  - `single_arm_direct`
  - `single_arm_with_support`
  - `bimanual_symmetric`
  - `bimanual_asymmetric_stabilize_manipulate`
  - `bimanual_sequential_handoff`
- 这条轴对后续双臂能力分析有必要保留；否则难以区分模型失败源于语义长尾还是源于协作结构本身。

4. 为什么当前不直接让 annotator 判 family
- family 边界是否稳定，前提是 `primitive_1` 和 primitive 集合本身先稳定。
- 如果过早要求 annotator 直接压缩到 family，会把“primitive 边界不清”伪装成“family 分歧”。
- 当前更需要先看：
  - `primitive_1` 是否稳定
  - primitive set overlap / jaccard 是否稳定
  - `coordination_pattern` / `common_vs_long_tail` 是否稳定
- 只有 pilot 证明 primitive 层一致性足够高时，family 聚合才值得进入下一步设计。

5. 当前标注策略
- 先不直接做“纯人工逐条随手填表”，也不做“LLM 一键自动打标”。
- 当前采用 `primitive-first` 的 human-in-the-loop 方案：
  - 先基于 task metadata 生成 `motion primitive candidates`
  - annotator 填写 `primitive_1 / primitive_2 / primitive_3`
  - annotator 再填写 `coordination_pattern / common_vs_long_tail / droid_overlap`
- pilot 阶段按双人独立标执行：
  - `primitive_1`
  - `primitive_2`
  - `primitive_3`
  - `coordination_pattern`
  - `common_vs_long_tail`
  - `droid_overlap`
  - `long_tail_rationale`
  - `notes`
- `semantic` 与 `affordance` 现在拆成两套并行 workflow：
  - `semantic` 当前围绕 `primitive layer / coordination / long-tail`
  - `affordance` 单独进入 `task_affordance_annotation_v1.csv`，不再混在 semantic 主表里
- pilot 结束后优先统计：
  - `primitive_1` agreement
  - primitive set overlap / mean jaccard
  - `coordination_pattern` 的 agreement / kappa
  - `common_vs_long_tail` 的 agreement / kappa
  - `droid_overlap` 只作为 sidecar
- family 聚合如果要做，应放在 pilot 之后，并且明确声明它是 benchmark-side derived layer。

6. 叙述口径约束
- 文档中不要写“采用 GM-100 官方 HAKE 标签”。
- 更安全的写法是：
  - GM-100 的任务设计受到 HAKE / HOI primitive / affordance 思想启发；
  - benchmark 侧当前新增的是一层 `primitive-first` task taxonomy，用于分析和审计；
  - 若后续报告 family aggregation，应明确说它是从 adjudicated primitives 派生出来的 benchmark-side analytic layer。
- `common_vs_long_tail` 应明确写成 benchmark-side analytic label，而不是数据集官方现成字段。

7. 当前进展（2026-04-02 更新）
- 已新增目录：
  - `benchmark/manual_audit/semantic_affordance_audit/`
- 已落地的 semantic 文件：
  - `benchmark/manual_audit/semantic_affordance_audit/build_task_semantic_seed.py`
  - `benchmark/manual_audit/semantic_affordance_audit/motion_primitive_candidates_v1.md`
  - `benchmark/manual_audit/semantic_affordance_audit/task_semantic_guideline_v1.md`
  - `benchmark/manual_audit/semantic_affordance_audit/README_FOR_SEMANTIC_ANNOTATOR.md`
  - `benchmark/manual_audit/semantic_affordance_audit/score_semantic_annotations.py`
  - `benchmark/manual_audit/semantic_affordance_audit/task_semantic_annotation_v1.csv`
  - `benchmark/manual_audit/semantic_affordance_audit/task_semantic_annotation_annotator_a_v1.csv`
  - `benchmark/manual_audit/semantic_affordance_audit/task_semantic_annotation_annotator_a_v1.xlsx`
- 已落地的 affordance 文件：
  - `benchmark/manual_audit/semantic_affordance_audit/build_task_affordance_template.py`
  - `benchmark/manual_audit/semantic_affordance_audit/import_gm100_list_inventory.py`
  - `benchmark/manual_audit/semantic_affordance_audit/README_FOR_AFFORDANCE_ANNOTATOR.md`
  - `benchmark/manual_audit/semantic_affordance_audit/task_affordance_annotation_v1.csv`
  - `benchmark/manual_audit/semantic_affordance_audit/task_affordance_annotation_annotator_a_v1.csv`
  - `benchmark/manual_audit/semantic_affordance_audit/task_affordance_annotation_annotator_a_v1.xlsx`
  - `benchmark/manual_audit/semantic_affordance_audit/task_object_inventory_v1.json`
- 已落地的 annotator package：
  - `benchmark/manual_audit/semantic_affordance_audit/annotator_packages/task_taxonomy_v1_annotator_a/`
  - 当前仓库快照只保留了 annotator A package；annotator B sidecar 尚未随这次路径重构同步检入
  - 当前已检入的 package 包含：
    - semantic workbook
    - affordance workbook
    - short guide
    - full guideline
    - primitive candidate list
- 已新增的叙述边界附录：
  - `task_semantic_guideline_v1.md` 文末已追加 DROID / AgiBot World 公开材料对照边界
  - 用途是后续 benchmark-side narrative / analysis reference
  - 不是 annotator 直接填写的字段定义，也不是外部数据集官方 primitive ground truth
- 当前 seed 来源：
  - `gm100-cobotmagic-lerobot/task_*/meta/tasks.jsonl`
  - `GM100_bimanual_fullscan_20260318/task_type_annotation.csv`
- 当前 semantic 覆盖范围：
  - 已成功生成 `106` 个 task 的 primitive seed 标注包
  - 其余 `4` 个 task 暂缺 `meta/tasks.jsonl`，因此未进入 semantic 首版 seed：
    - `task_00004`
    - `task_00005`
    - `task_00052`
    - `task_00059`
- 当前 semantic CSV 不是空模板，而是带有以下自动 seed 列：
  - `task_id`
  - `task_name_raw`
  - `task_name_readable`
  - `arm_type`
  - `primary_arm`
  - `source_meta_path`
  - `primitive_candidates_auto`
  - `coordination_candidates_auto`
- semantic annotator 需要人工填写的主字段当前为：
  - `primitive_1`
  - `primitive_2`
  - `primitive_3`
  - `coordination_pattern`
  - `common_vs_long_tail`
  - `droid_overlap`
  - `long_tail_rationale`
  - `notes`
- 当前 affordance 覆盖范围：
  - affordance 主表已覆盖 `110` 个 task 位点
  - 其中 `106` 个是 `metadata_ready`
  - `4` 个是 `missing_task_metadata`
  - `GM100 List.xlsx` 已导入 `object_inventory_raw`
  - 当前 inventory 覆盖 `108/110` 个 task
  - 当前缺 inventory 的 task：
    - `task_00059`
    - `task_00089`
- 当前 affordance ontology 已固定为 4 个字段轴：
  - `object_affordance_tags`
  - `interface_affordance_tags`
  - `constraint_affordance_tags`
  - `role_tags`
- 当前 annotate guideline 已做一致性补齐（已完成）
  - `README_FOR_SEMANTIC_ANNOTATOR.md`
  - `README_FOR_AFFORDANCE_ANNOTATOR.md`
  - `task_semantic_guideline_v1.md`
  - 已同步说明实际发包时优先使用 `benchmark/manual_audit/semantic_affordance_audit/annotator_packages/` 内的 package 版本
- 当前明确尚未做的事：
  - 还没有产出全量 `affordance seed candidates`
  - 还没有开始 affordance dual-annotation pilot
  - 还没有开始 semantic dual-annotation pilot
  - 还没有进入 family aggregation 设计与裁决

8. 下一步具体动作
- semantic 方向：
  - 从当前 `106` 个 seed task 中选取约 `20` 个 task 做 semantic pilot annotation
  - pilot 需覆盖：
    - common / long-tail
    - single-arm / bimanual
    - open-access / insert-attach / transfer / tool-mediated / deformable / dynamic-force 等主要 primitive 区域
  - 两位 annotator 按 `task_semantic_guideline_v1.md` 独立填写 A/B 两份 semantic CSV
  - 先看冲突是否主要集中在：
    - `primitive_1`
    - primitive set overlap
    - `coordination_pattern`
    - `common_vs_long_tail`
- affordance 方向：
  - 基于 `object_inventory_raw + task_name_raw` 先生成一版 affordance seed candidates
  - 再选 `15-20` 个 task 做 affordance pilot
  - pilot 重点看：
    - `object_affordance_tags`
    - `interface_affordance_tags`
    - `constraint_affordance_tags`
    - `role_tags`
- 根据 semantic / affordance 两个 pilot 的冲突情况，再决定是否需要补 primitive 词表、改 primitive rule、或重写 affordance ontology 边界。
- family aggregation 是否进入下一步，取决于 primitive pilot 的一致性结果，而不是先验拍板。

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
- 这样做的目的：保持 `eval_v1/run_benchmark_v1_eval.py` 与 `manual_audit/gt_audit/export_audit_subset.py` 仍可继续使用同一个 root frame dir。

4. manual audit 工具链当前已可用（已完成）
- 标注说明（简体中文）：`benchmark/manual_audit/gt_audit/audit_guideline_v1.md`
- 标注模板：`benchmark/manual_audit/gt_audit/audit_template_v1.csv`
- 导出脚本：`benchmark/manual_audit/gt_audit/export_audit_subset.py`
- 评分/汇总脚本：`benchmark/manual_audit/gt_audit/score_audit_annotations.py`
- 当前 card 形式：每道题会把“图片/多帧 + Task Meta + 问题 + 选项 + benchmark GT”合成在同一张 `audit card` JPG 中；annotator 只需看 card 并把判断填回 CSV。

5. 历史 pilot audit 包当前不再保留在仓库中
- 先前的 full-task pilot 与 boundary-only pilot 都已移除，不再作为当前工作目录。
- 如果后续还需要新的 pilot / smoke test，应使用：
  - `python benchmark/manual_audit/gt_audit/export_audit_subset.py --output-dir benchmark/manual_audit/<your_pilot_dir>`
- 当前仓库内 manual audit 的唯一活跃目录是：
  - `benchmark/manual_audit/gt_audit/full_audit_v1/`

6. 当前已经确认的风险（需要后续审计验证）
- “前后 1 秒过滤”只能解决一部分 FOV 问题，不能保证样本一定可答。
- 目前已观察到：即使不在视频开头/结尾，top-view 里仍可能看不到机械臂，但底层物理信号仍然存在；这会继续影响 `T3/T4/T6`，以及部分 `T_temporal/T_binary` 的可答性。
- 因此：
  - 当前 1s guard 应视为 `simple first-pass filter`
  - 不能把它当作最终的 visibility validity solution

7. pilot 的当前口径
- pilot 现在不再依赖仓库内预置目录。
- 如果需要新的协议 smoke test 或 targeted diagnostics，应临时导出并显式命名，而不是默认复用历史 `pilot_audit_*` 目录。

8. 当前 manual audit 的工作约定
- 默认工作目录统一为：`benchmark/manual_audit/gt_audit/full_audit_v1/`
- 任何新的 pilot / targeted pack 都应视为临时 sidecar，按需导出、单独命名、用完即可归档或删除。

9. 正式版 full audit 包已生成（已完成）
- 路径：`benchmark/manual_audit/gt_audit/full_audit_v1/`
- 题型与配额：
  - `T1=80`
  - `T2=60`
  - `T3=60`
  - `T4=60`
  - `T6=60`
  - `T_binary=50`
  - `T_progress=50`
  - `T_temporal=50`
- 总量：`470`
- 已生成：
  - `audit_summary.json`
  - `audit_items.csv`
  - `annotator_a.csv`
  - `annotator_a.xlsx`
  - `audit_subset.jsonl`
  - `audit_cards/*.jpg`
- 注意：
  - 这版 formal audit 现已把 `T_binary` 纳入正式审计，而不是只留在 sidecar pilot。
  - 当前正式审阅统一依赖 `audit_cards/*.jpg`；`T_binary` 的比较 panel 已嵌入卡片，不再单列 `review_assets/`。
  - 原始 `420` 条 7-family formal audit 仍保留原顺序；`T_binary=50` 作为正式增补追加到 `full_audit_v1/`。
  - 原始 7-family 导出时 `relaxed_task_cap=true`，说明为了凑满 `420` 条，`max_per_task_id=2` 的软限制被放松过；后续追加的 `T_binary=50` 未触发 cap 放松。

10. `full_audit_v1` 已做完整性核对（已完成）
- 已核对通过：
  - `audit_cards/` 实际文件数 = `470`
  - `audit_items.csv` 数据行数 = `470`
  - `annotator_a.csv` 数据行数 = `470`
  - `audit_subset.jsonl` 行数 = `470`
  - `audit_summary.json.selected_total = 470`
  - `audit_items.csv` / `annotator_a.csv` / `audit_subset.jsonl` 的 `audit_item_id` 与 `item_index` 顺序一致
  - 全部 `audit_card_path` 均存在
  - 全部 `frame_paths` 均存在

11. `full_audit_v1` audit cards 已追加 task meta（已完成）
- 生效位置：
  - `benchmark/manual_audit/gt_audit/export_audit_subset.py`
- task meta 来源：
  - 优先读取 `benchmark/GM100 List.xlsx` 的 `Task Description`
  - 若 xlsx 缺失，则 fallback 到 dataset `tasks.jsonl` 的 task 名可读化
- 本次更新方式：
  - 不重新采样，不改 `audit_subset.jsonl`
  - 直接按现有 `full_audit_v1/audit_subset.jsonl` 顺序重刷并覆盖 `audit_cards/*.jpg`
- 覆盖结果：
  - 当前 formal audit 已覆盖 `470` 张 card，其中 `T_binary` 所需比较 panel 已直接嵌入 `audit_cards/*.jpg`
  - 形式上新增 `Task Meta` box
  - 当前 `full_audit_v1` 中没有 card 缺 task meta 文本
  - `task_00089` 虽然不在 xlsx 中，但已通过 dataset fallback 显示为 `disassemble luban lock`

12. 现在最应该做的事（next）
- 以 `benchmark/manual_audit/gt_audit/full_audit_v1/` 作为正式人工审计工作目录，开始双人独立标注。
- 标完后用 `benchmark/manual_audit/gt_audit/score_audit_annotations.py` 计算：
  - overall agreement
  - per-task agreement
  - `gt_correctness` 的一致性
  - 主要 error category 分布
- 如果正式 audit 中 `visual_answerability=not_answerable` 在 `T3/T4/T6/T_temporal/T_binary` 中仍显著偏高，则下一步不要只继续调 head/tail 秒数，而是要加第二层 `visibility / answerability gate`。
- 在 formal audit 结果出来前，不建议把当前 root benchmark 宣称为“visibility issue 已解决”；更准确的说法应是：
  - 已做了基于 episode 边界的 first-pass filtering；
  - formal manual audit 已启动，用于验证剩余样本的可回答性与 GT 有效性。

### AI. Eval Prompt Task Meta 开关（2026-04-01）

1. 当前实现状态（已完成）
- 已在：
  - `benchmark/eval_v1/run_pilot_eval.py`
  中新增两个参数：
  - `--prepend-task-meta`
  - `--task-meta-xlsx`
- `benchmark/eval_v1/run_benchmark_v1_eval.py` 因为复用 `run_pilot_eval.py` 的 parser 与执行逻辑，现已自动继承这两个参数。

2. 行为定义（已完成）
- 当 `--prepend-task-meta` 关闭时：
  - prompt 行为与之前一致，不影响已有 benchmark 复现。
- 当 `--prepend-task-meta` 开启时：
  - 每个问题的 prompt 前面会先加入一段 task-level episode context
  - 默认从 `benchmark/GM100 List.xlsx` 读取 task meta
  - 若 xlsx 缺该 task，则 fallback 到 dataset `tasks.jsonl`
- 当前 prompt 口径强调：
  - task meta 只是 `background`
  - 不能仅凭 task name 作答
  - 回答必须基于当前图片/多帧中的 visual evidence
- 当前代码中的精确 prepend 模板为：

```text
You are given image(s) from a robot manipulation episode.

Task context: The overall task in this episode is "<task_meta>".

This task context is provided only as background. Do not rely on the task name alone. Answer based on the visual evidence in the provided image(s).

Now answer the following question:
<prompt_body>
```

- 其中 `<prompt_body>` 指原本该题自己的 question + choices / answer-format 指令块。

3. 当前用途判断
- 这不是 root benchmark 默认设定，而是一个可选的 prompt-ablation / prompt-augmentation 开关。
- 后续如果开启此开关跑实验：
  - 结果必须和“无 task meta prompt”的主结果分开记录
  - 不能直接混写成同一条 benchmark 主结果

4. 下一步
- 若要正式使用这条设置做实验，建议：
  - 固定一个明确的输出命名规范（例如 `*_with_taskmeta.jsonl`）
  - 单独报告 `prepend_task_meta=true` 的结果
  - 与默认 prompt 做 paired comparison，而不是替代默认主结果

### AJ. 文档分工清理（2026-04-02）

1. 本次调整的目标（已完成）
- 不再让 `NEXT_STEPS.md` 继续承担 architecture map / quickstart README 的职责。
- 文档分工收敛为：
  - `README.md`：最常用命令
  - `ARCHITECTURE_MAP.md`：当前稳定结构、数据流、默认入口
  - `NEXT_STEPS.md`：近期决策、风险判断、优先级与 next actions
  - `benchmark_card.md`：对外叙述与验证状态

2. 本次已执行的整理（已完成）
- 已将 `NEXT_STEPS.md` 前半段的重复目录结构、数据流、脚本 I/O 和大段运行命令移出。
- 已把 `ARCHITECTURE_MAP.md` 扩展为覆盖三条当前主线：
  - `benchmark_v1`
  - `manual_audit`
  - `task_taxonomy`
- 已在 `ARCHITECTURE_MAP.md` 中显式纳入：
  - `GM100 List.xlsx`
  - `manual_audit/`
  - `manual_audit/semantic_affordance_audit/`
  - `run_v1_pipeline.sh`
  - `manual_checks_20260325/` 的 sidecar 定位

3. 后续维护约束
- 如果只是更新默认入口、目录结构、数据流、默认 frame/source snapshot，应优先改 `ARCHITECTURE_MAP.md`。
- 如果只是更新“现在最该做什么”、方法论风险、pilot 观察或 superseding note，应优先改 `NEXT_STEPS.md`。
- 只有当 benchmark 对外叙述、validation status 或 caveat 发生变化时，才同步改 `benchmark_card.md`。

### AK. `qwen25vl32b` task-meta / instruction 对比结论（2026-04-03）

1. 本次比较对象与口径
- 比较文件：
  - `benchmark/eval_results_v1/benchmark_v1_qwen25vl32b_full.jsonl`
  - `benchmark/eval_results_v1/benchmark_v1_qwen25vl32b_full_new.jsonl`
- 当前工作假设：
  - `full` = 默认 benchmark prompt
  - `full_new` = 加入 task-goal / instruction anchoring 的 rerun
- 说明：
  - 结果 JSONL 本身没有显式记录 `prepend_task_meta=true/false`
  - 因此这次结论依赖当前文件命名与运行上下文；后续若继续做这条分析，输出文件应改成显式命名（例如 `*_with_taskmeta.jsonl`）

2. overall gap
- `full`: `30.83%`
- `full_new`: `34.46%`
- overall gap: `+3.63pp`
- item-level paired flip：
  - `wrong -> correct`: `1252`
  - `correct -> wrong`: `689`
  - net gain: `+563`
- 当前最重要的判断不是“整体升了 `3.63pp`”，而是：
  - instruction gain **高度不均匀**
  - 增益主要被少数 capability / task family 拉动
  - 不能把它叙述成“所有题型都稳定受益”

3. per-task-type gap
- `T3`: `13.36% -> 33.84%`，`+20.48pp`
- `T2`: `62.30% -> 65.60%`，`+3.30pp`
- `T_progress`: `18.04% -> 20.88%`，`+2.84pp`
- `T_binary`: `47.80% -> 49.07%`，`+1.27pp`
- `T6`: `63.73% -> 64.53%`，`+0.80pp`
- `T_temporal`: `15.90% -> 15.65%`，`-0.25pp`
- `T1`: `31.40% -> 29.67%`，`-1.73pp`
- `T4`: `29.13% -> 27.33%`，`-1.80pp`

4. 这组对比最值得保留的主发现
- **哪些题型最依赖 instruction**
  - 明显是 `T3`
  - `T3` 单题型贡献了这次 net gain 的绝大部分，说明 task-goal / instruction 最能帮助的是：
    - 已知任务目标后，对当前主运动方向 / action intent 的判别
- **哪些题型几乎不受 instruction 影响**
  - `T6`
  - `T_binary`
  - `T_temporal`
  - 说明 task context 不能替代真实的 temporal evidence；对纯排序 / 先后判断帮助很有限
- **哪些题型反而被 instruction 干扰**
  - `T1`
  - `T4`
  - 当前解释更像是：
    - goal prior 在某些 state-reading / explicit arm-state 题上会引入额外偏置
    - 不应默认认为“知道任务目标”一定帮助所有题型

5. per-capability-dimension gap（按题型能力束）
- `motion_state_direction`（`T3 + T6`）：
  - `32.25% -> 45.35%`，`+13.10pp`
- `single_frame_static`（`T1 + T2 + T_progress`）：
  - `31.02% -> 31.82%`，`+0.80pp`
- `temporal_ordering`（`T_temporal + T_binary`）：
  - `29.57% -> 29.97%`，`+0.40pp`
- `bimanual_state_reasoning`（`T4`）：
  - `29.13% -> 27.33%`，`-1.80pp`
- 当前可直接写进 paper 的结论：
  - instruction gain 的主来源不是 general boost
  - 而是明显集中在 `motion direction / action intent` 相关能力
  - 对 `temporal ordering` 几乎没有帮助

6. per-capability-dimension gap（按 primitive seed capability）
- 增益较大的 primitive 区域包括：
  - `carry_transport`: `+7.75pp`
  - `hang_hook_mount`: `+7.14pp`
  - `push_pull_drag`: `+6.29pp`
  - `handover_feed_present`: `+6.09pp`
  - `shake_agitate_mix`: `+5.79pp`
  - `fold_wrap_braid`: `+5.72pp`
  - `search_retrieve_inspect`: `+5.21pp`
- 几乎不受 instruction 影响的 primitive 区域包括：
  - `twist_rotate_unlock`: `-0.39pp`
  - `cover_hide_block`: `+0.37pp`
  - `insert_plug_skewer`: `+0.37pp`
  - `cut_slice_divide`: `+1.50pp`
- 当前解释：
  - task-goal / instruction 更像是在提供 **goal-conditioned action prior**
  - 对 `push / carry / handover / search` 这类“目标会约束动作意图”的任务帮助更大
  - 对 `insert / twist / cut` 这类更依赖局部几何与精细视觉证据的任务帮助有限

7. common vs long-tail gap（当前仅为 proxy，不是正式 taxonomy 结论）
- 由于 `task_taxonomy` 的人工 `common_vs_long_tail` 还未完成，这里当前只能使用 proxy：
  - 按 task 的 `primary primitive seed` 在全部 task 中的出现频次做二分
  - 频次 `<=4` 记为 `long_tail_proxy`
  - 频次 `>4` 记为 `common_proxy`
- 结果：
  - `common_proxy`: `30.57% -> 34.37%`，`+3.80pp`
  - `long_tail_proxy`: `31.09% -> 34.55%`，`+3.46pp`
- 当前判断：
  - **没有证据表明 long-tail task 更依赖 instruction anchoring**
  - 至少在当前 proxy 下，long-tail gain 并不比 common 更强
- 口径约束：
  - 这条只能先作为 interim analytic observation
  - 等 `task_taxonomy` 的正式 `common_vs_long_tail` 标签完成后，再决定是否升级为 headline claim

8. bimanual coordination 是否在 knowing-the-goal 后提升更明显
- 粗粒度对比：
  - `bimanual_combined`: `30.73% -> 35.13%`，`+4.40pp`
  - `single_combined`: `31.49% -> 35.38%`，`+3.90pp`
- coordination seed 对比：
  - `bimanual_sequential_handoff`: `+4.28pp`
  - `single_arm_direct`: `+3.90pp`
  - `bimanual_asymmetric_stabilize_manipulate`: `+3.50pp`
  - `bimanual_symmetric`: `+2.40pp`
- `T3` 内部进一步看：
  - `bimanual_sync`: `+21.12pp`
  - `bimanual_sequential`: `+20.63pp`
  - `single_left`: `+20.09pp`
  - `single_right`: `+19.91pp`
- 当前更准确的说法是：
  - **bimanual 整体比 single-arm 更吃 instruction，但优势不大**
  - 且这个效应主要来自 `T3` 这类 action-intent / motion-direction 题
  - 并不是所有 bimanual reasoning 都更受益；`T4` 反而下降

9. 对论文主发现的直接启发
- 更好的 headline 不是：
  - “task instruction 让整体 accuracy 上升 `3.63pp`”
- 更值得写成主发现的是：
  - task-goal / instruction anchoring 的收益 **高度集中在 motion-direction / action-intent reasoning**
  - 它对 `temporal ordering` 几乎无帮助
  - 它对部分 state-reading / arm-state reasoning 题甚至会引入偏置
  - long-tail 是否更吃 instruction，目前 **没有支持证据**
  - bimanual 相比 single-arm 有轻微更高的 instruction gain，但主要是被 `T3` 拉动，而不是通用规律

10. 下一步
- 若继续保留这条分析线，必须修正输出命名：
  - 默认 prompt：`*_full.jsonl`
  - task-meta / instruction 版：`*_with_taskmeta.jsonl`
- 后续 paired comparison 应作为单独小节汇报：
  - overall
  - per-task-type
  - per-capability-dimension
  - common vs long-tail
  - bimanual vs single-arm
- 在 `task_taxonomy` 的人工标签完成前：
  - `common vs long-tail` 只能写成 proxy analysis
  - 不应把它写成正式 benchmark-side confirmed label result
