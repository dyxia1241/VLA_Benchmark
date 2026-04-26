# NeurIPS E&D Minimal Submission Plan

## 1. Thesis

论文主线：

> A goal-conditioned, signal-grounded, process-aware benchmark for robotic manipulation.

核心对象不是单一 primitive taxonomy，也不是单一 contact signal，而是机器人操作过程理解：

- 当前处于什么过程状态
- 接下来会发生什么
- 局部操作完成到哪一步
- 动态结果与失败风险如何判断

`primitive`、`contact`、`progress`、`failure` 都是 grounding mechanism。标题和摘要层面统一使用 `process-aware`。

## 2. Minimal Closed Loop

投稿版最小闭环已经收敛到四件事：

1. 定义 process-aware benchmark topology。
2. 用 `GM-100 + RH20T + REASSEMBLE + AIST` 构建四源题库。
3. 产出严格 `episode-level` 隔离的 `SFT / Eval` split。
4. 提供一键运行全部 eval set 的执行与汇总入口。

当前主文不再依赖 `RoboMIND2.0` 才成立。`RoboMIND2.0` 只保留为后续扩展源。

## 3. Benchmark Topology

### 3.1 Two Families

| Family | Goal | Tasks |
| --- | --- | --- |
| Static process monitoring | 判断当前状态、接触、协同与当前过程标签。 | `T1 / T2 / T4 / T10` |
| Dynamic process reasoning | 判断运动方向、局部进度、结果、时序和后续步骤。 | `T3 / T5 / T6 / T7 / T8 / T9 / T11 / T12` |

命名约定：

- `T5 == T_progress`
- `T8 == T_temporal`
- `T9 == T_binary`

工程文件可以保留旧名，论文统一使用 `T5 / T8 / T9`。

### 3.2 Twelve Tasks

| ID | Name | Core Question | Input | Main Sources |
| --- | --- | --- | --- | --- |
| `T1` | Phase Recognition | 当前处于哪个粗粒度过程阶段？ | 单帧 | GM-100, RH20T, REASSEMBLE |
| `T2` | Contact Detection | 当前是否已发生有效接触？ | 单帧 | GM-100, RH20T, REASSEMBLE |
| `T3` | Motion Direction Prediction | 主要运动方向是什么？ | 短时序帧 | GM-100, RH20T, AIST |
| `T4` | Bimanual Coordination State | 双臂当前如何协同？ | 短时序帧 | GM-100, AIST |
| `T5` | Primitive-local Progress | 当前 local step 处于 early / middle / late 哪段？ | 短时序帧 | GM-100, RH20T, REASSEMBLE |
| `T6` | Motion State Recognition | 当前是否处于明显运动状态？ | 短时序帧 | GM-100, RH20T, REASSEMBLE, AIST |
| `T7` | Operation Outcome Prediction | 给定操作早期片段，最终会成功还是失败？ | 拉长间隔多帧 | RH20T, REASSEMBLE |
| `T8` | Temporal Ordering | 三个片段的真实时间顺序是什么？ | 三帧乱序 | GM-100, RH20T, REASSEMBLE, AIST |
| `T9` | Temporal Priority Prediction | 两个片段哪个先发生？ | 双 panel / 双帧 | GM-100, RH20T, REASSEMBLE, AIST |
| `T10` | Current Primitive Recognition | 当前正在执行哪个低层动作？ | 短时序帧 | REASSEMBLE |
| `T11` | Next Primitive Prediction | 给定当前过程，下一个动作是什么？ | 短时序帧 + context | REASSEMBLE |
| `T12` | Primitive Chain Restoration | 完成任务所需动作链如何复原？ | 局部链条 / mask | REASSEMBLE |

## 4. Shared Primitive Definition

共享 primitive 闭集采用 coarse process primitives：

| Primitive | Meaning |
| --- | --- |
| `engage` | 建立有效接触、抓取、按住、接管对象 |
| `stabilize` | 保持、压住、支撑、固定对象 |
| `transport` | 搬运、推拉、转移对象位置 |
| `align` | 对准、定位、姿态调整、插入前配准 |
| `effect` | 真正改变任务状态的动作提交 |
| `disengage` | 松开、释放、撤离、结束交互 |
| `failure` | 明确偏离目标、掉落、未完成、错误接触 |

`reach/approach` 不进入主闭集。它更像接触前的必然过渡，信息收益低，噪声高。

人工标注口径保持最小化：

- 标 `task-level primitive chain`
- 标 `object slot`
- 不做全量逐帧 primitive 标注

## 5. Dataset Roles

| Dataset | Role | Supported Tasks | Current Status |
| --- | --- | --- | --- |
| `GM-100` | 主数据源；goal-conditioned；覆盖单臂与双臂 | `T1, T2, T3, T4, T5, T6, T8, T9` | 已冻结 `15,500` 题 |
| `RH20T` | force/torque + gripper + TCP 驱动的 process/contact 数据源 | `T1, T2, T3, T5, T6, T7, T8, T9` | 已冻结 `15,800` 题，`596` scenes，`76` task IDs |
| `REASSEMBLE` | 最干净的 action-chain / outcome 数据源 | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` | 已冻结 `17,165` 题，`37` recordings |
| `AIST-Bimanual` | 无力信号的双臂运动学数据源 | `T3, T4, T6, T8, T9` | 已冻结 `9,427` 题，`200` episodes |

覆盖原则：

- 不要求每个数据集覆盖全部 `12` 题
- 每个数据集只承担其信号稳定支持的题型

## 6. Current Release Snapshot

### 6.1 Full Curated Benchmark

| Source | Curated JSONL | Frame Cache | Eval Wrapper |
| --- | --- | --- | --- |
| GM-100 | `benchmark/benchmark_v1_curated.jsonl` | `benchmark/benchmark_v1_frames_tbinary_20260330/` | `benchmark/eval_v1/run_benchmark_v1_eval.py` |
| RH20T | `benchmark/rh20t_benchmark_v0_curated.jsonl` | `benchmark/rh20t_benchmark_v0_frames/` | `benchmark/eval_v1/run_rh20t_benchmark_v0_eval.py` |
| REASSEMBLE | `benchmark/reassemble_benchmark_v0_curated.jsonl` | `benchmark/reassemble_benchmark_v0_frames/` | `benchmark/eval_v1/run_reassemble_benchmark_v0_eval.py` |
| AIST | `benchmark/aist_benchmark_v0/aist_benchmark_v0_curated.jsonl` | `benchmark/aist_benchmark_v0_frames/` | `benchmark/eval_v1/run_aist_benchmark_v0_eval.py` |

### 6.2 Current Full Counts

| Source | Items | Coverage |
| --- | ---: | --- |
| GM-100 | `15,500` | `T1, T2, T3, T4, T5, T6, T8, T9` |
| RH20T | `15,800` | `T1, T2, T3, T5, T6, T7, T8, T9` |
| REASSEMBLE | `17,165` | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` |
| AIST | `9,427` | `T3, T4, T6, T8, T9` |

当前四源全量合计：`57,892` items。

## 7. SFT / Eval Split

### 7.1 Split Rule

当前已经构建严格的 `episode-level` 隔离 split：

- `85%` SFT
- `15%` Eval
- 同一个 `episode / recording / scene` 只会出现在一个 split 中

split 产物目录：

- `benchmark/splits_v1/`

关键文件：

- `benchmark/splits_v1/all_sft_merged.jsonl`
- `benchmark/splits_v1/all_eval_merged.jsonl`
- `benchmark/splits_v1/split_summary.json`
- `benchmark/splits_v1/eval_manifest.json`

### 7.2 Current Split Counts

| Source | Total | Eval | SFT | Group Unit | Eval Groups |
| --- | ---: | ---: | ---: | --- | ---: |
| GM-100 | `15,500` | `2,643` | `12,857` | `(task_id, episode_id)` | `628` |
| RH20T | `15,800` | `2,422` | `13,378` | `recording_id` | `89` |
| REASSEMBLE | `17,165` | `2,562` | `14,603` | `recording_id` | `6` |
| AIST | `9,427` | `1,424` | `8,003` | `recording_id` | `30` |

合计：

- Eval：`9,051`
- SFT：`48,841`

### 7.3 Split Design Choice

`REASSEMBLE` 的 `T10 / T11 / T12` 最脆弱，因此 split 不是纯随机。  
当前 `splits_v1` 对 `REASSEMBLE` 的 `recording-level` 选择显式提高了 `T10 / T11 / T12` 的权重，避免 eval 中这三类题被稀释。

## 8. GT Construction Rules

1. 优先使用 dataset-native signal：force/torque、gripper state、TCP、joint state、velocity、official segment labels、task metadata。
2. 每个题型先定义 local unit，再采样 frame。
3. GT builder 自动生成，人工只做 audit 和 taxonomy 校准。
4. `task_meta` 是可控变量；主结果需明确是否启用。
5. `primitive-object pair` 用作辅助标注与分析，不升级为统一主 GT。

### 8.1 T5 / T_progress v2

`T5` 统一定义为 within-local-step progress。

GT 连续值：

`u = (t - start(P_k)) / (end(P_k) - start(P_k))`

发布标签：

- `early`
- `middle`
- `late`

对应 local step 来源：

| Dataset | `P_k` source |
| --- | --- |
| GM-100 | contact-aware local interaction interval |
| RH20T | force/torque + gripper + TCP 合成 local interval |
| REASSEMBLE | 官方 successful high-level segment |

## 9. Evaluation Protocol

### 9.1 Default Eval

当前推荐协议不是把四源题库硬合成一个共享 frame root 的大 JSONL，而是：

1. 四个 dataset 保持各自 eval JSONL
2. 用统一总控脚本顺序调四个 dataset wrapper
3. 再统一汇总结果

一键运行入口：

- `benchmark/eval_v1/run_all_eval_sets.py`
- `benchmark/eval_v1/score_all_eval_sets.py`

### 9.2 What One-click Eval Produces

默认结果目录：

- `benchmark/eval_results_v1/splits_v1/`

运行后会生成：

- `gm100_eval_results.jsonl`
- `rh20t_eval_results.jsonl`
- `reassemble_eval_results.jsonl`
- `aist_eval_results.jsonl`
- `run_all_eval_sets_summary.json`
- `score_all_eval_sets_summary.json`

### 9.3 Why This Design

原因很简单：

- 四个数据集现在分属四套 frame cache
- 现有通用执行器 `run_pilot_eval.py` 只有一组 frame-dir 参数
- 用 manifest + dataset-specific wrapper 更稳，也更容易复现

## 10. Manual Audit

人工标注员：Huiting Ji。

人工工作只保留两类：

1. `benchmark validity audit`
2. `task-level primitive-object chain audit`

不做三件事：

1. 不做全量逐帧 primitive 标注
2. 不做全量 object grounding 标注
3. 不把 semantic verb-object pair 当成统一 benchmark 主标签

当前高优先级 audit：

- `T5`
- `T7`
- `T10`
- `T11`
- `T12`
- `T3` 方向映射校准

## 11. Current Risks

1. `REASSEMBLE` 只有 `37` 个 recording，因此 `T10/T11/T12` 的 eval 统计仍然比大规模题型更脆弱。
2. `AIST` 当前没有力信号，因此它承担的是双臂运动学与时序题，不承担 contact / failure 主证据。
3. `RH20T` 的视角协议已经固定到主相机，但 viewpoint 仍应在 paper 中作为 caveat 明确写出。
4. `RoboMIND2.0` 尚未进入主文最小闭环，不应写成已经完成的主结果来源。

## 12. Immediate Next Actions

| Priority | Task | Exit Condition |
| --- | --- | --- |
| P0 | 把 `splits_v1` 与 one-click eval 写入主文档与 benchmark card | 文档口径统一 |
| P0 | 跑四源 eval baseline | 产出 `score_all_eval_sets_summary.json` |
| P1 | 用 `all_sft_merged.jsonl` 准备单模型 SFT | train/eval protocol 固定 |
| P1 | 完成高风险题型人工审计统计 | 可写入 paper reliability section |
| P2 | 将 RoboMIND2.0 作为附录扩展源接入 | 不影响主文闭环 |

## 13. One-line Positioning

> We present a goal-conditioned, signal-grounded benchmark for robotic manipulation process understanding, centered on phase recognition, contact state, motion direction, primitive-local progress, temporal reasoning, failure-aware outcome prediction, and action-chain understanding across four heterogeneous robot datasets.
