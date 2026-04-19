# NeurIPS E&D Minimal Submission Plan

## 1. Paper Thesis

本文主线：构建一个 **goal-conditioned, signal-grounded, process-aware benchmark for robotic manipulation**。

核心评测对象是机器人操作过程理解：当前处于什么过程状态、接下来会发生什么、局部操作完成到哪一步、失败或时序风险在哪里。

`primitive`、`contact`、`progress`、`failure` 都是 grounding mechanism。标题和摘要层级统一使用 `process-aware`，避免把论文拖入单一 primitive 数据集或 contact 数据集赛道。

投稿最小闭环：

1. 定义 `2` 类、`12` 题的 process-aware benchmark topology。
2. 从 `GM-100 / REASSEMBLE / RH20T / AIST / RoboMIND2.0` 构建非对称覆盖的多源题库。
3. 做跨模型、跨题型、跨数据源评测。
4. 产出 curated supervision set。
5. 对一个 open-weight VLM 做 SFT。
6. 比较 base vs SFT，验证 benchmark-derived supervision 是否提升 process-aware 能力。

目标全量发布规模：`~100k` questions。投稿版先保证可复现、可审计、可评测的最小闭环成立。

## 2. Contributions

| 贡献 | 投稿表述 | 证据形式 |
| --- | --- | --- |
| Benchmark | 提出面向机器人操作过程理解的 goal-conditioned, signal-grounded benchmark。 | `12` 题 topology、统一 JSONL schema、frame cache、benchmark card。 |
| Multi-source GT builders | 用 dataset-native signal 和 metadata 自动构建 GT，减少逐帧人工标注。 | GM-100、REASSEMBLE、RH20T、AIST、RoboMIND2.0 的非对称 coverage。 |
| Evaluation analysis | 系统比较 VLM 在静态状态监测与动态过程推理上的能力差异。 | cross-model、cross-task、cross-source、with/without meta 分析。 |
| Supervision + SFT | 用 benchmark-derived data 训练一个 open-weight VLM，并和 base model 对比。 | base vs SFT 主表、held-out source / held-out task 分析。 |
| Audit protocol | 用小规模人工审计验证自动 GT 的可靠性。 | Huiting audit cards、分层抽检、错误类型统计。 |

## 3. Benchmark Topology

Benchmark 分成两类。

| 类别 | 目标 | 题型 |
| --- | --- | --- |
| Static process monitoring | 判断当前帧或短窗口内的可观测状态。 | `T1 / T2 / T4 / T10` |
| Dynamic process reasoning | 判断运动趋势、局部进度、结果、时序和下一步。 | `T3 / T5 / T6 / T7 / T8 / T9 / T11 / T12` |

### 3.1 Twelve Tasks

| ID | Name | 核心问题 | 输入 | 主要数据源 |
| --- | --- | --- | --- | --- |
| `T1` | Phase Recognition | 当前处于哪个粗粒度操作阶段？ | 单帧 | GM-100, REASSEMBLE, RH20T |
| `T2` | Contact Detection | 当前是否发生接触或交互？ | 单帧 | GM-100, REASSEMBLE, RH20T |
| `T3` | Motion Direction Prediction | 接下来主要往哪个方向运动？ | 短时序帧 | GM-100, AIST, RoboMIND2.0；RH20T pilot |
| `T4` | Bimanual Coordination State | 双臂当前如何协同？ | 短时序帧 | GM-100, AIST, RoboMIND2.0 |
| `T5` | Primitive-local Progress Recognition | 当前 local step 完成到 early / middle / late 哪一段？ | 短时序帧 | GM-100, REASSEMBLE, RH20T |
| `T6` | Motion State Recognition | 当前是否处于明显运动状态？ | 短时序帧 | GM-100, REASSEMBLE, RH20T, AIST, RoboMIND2.0 |
| `T7` | Operation Outcome Prediction | 给定操作早期片段，最终是否达成目标结果？ | 拉长间隔多帧 | REASSEMBLE, RH20T, AIST, RoboMIND2.0 |
| `T8` | Temporal Ordering | 三个片段的真实时间顺序是什么？ | 三帧乱序 | GM-100, REASSEMBLE, RH20T, RoboMIND2.0 |
| `T9` | Temporal Priority Prediction | 两个片段哪个先发生？ | 双 panel | GM-100, REASSEMBLE, RH20T, AIST, RoboMIND2.0 |
| `T10` | Current Primitive Recognition | 当前正在执行哪个 coarse primitive？ | 短时序帧 | GM-100, REASSEMBLE, RH20T |
| `T11` | Next Primitive Prediction | 给定当前过程，下一个 primitive 是什么？ | 短时序帧 + goal/meta | GM-100, REASSEMBLE, RH20T |
| `T12` | Primitive Chain Restoration | 完成任务所需的 primitive chain 如何复原？ | 局部链条 / mask | GM-100, REASSEMBLE, RH20T |

命名约束：`T5 == T_progress`，`T9 == T_binary`。工程文件可以保留旧名，论文统一使用 `T5 / T9`。

## 4. Shared Primitive Definition

共享 primitive 闭集采用 coarse process primitives，用于跨数据集对齐和人工审计。

| Primitive | 含义 | 典型对象槽 |
| --- | --- | --- |
| `engage` | 建立有效接触、抓取、按住、夹持或接管对象。 | tool / object / handle / button |
| `stabilize` | 保持、压住、支撑、固定对象，使后续动作可执行。 | object / container / surface |
| `transport` | 搬运、移动、推拉、转移对象位置。 | moved object / target region |
| `align` | 对准、插入前定位、姿态调整、配准。 | object / receptacle / slot |
| `effect` | 真正改变任务状态的操作提交。 | target object / mechanism |
| `disengage` | 松开、释放、撤离、结束当前交互。 | object / target state |
| `failure` | 明确偏离目标、掉落、未完成或错误接触。 | failed object / failed step |

`reach/approach` 不进入主闭集。它通常是接触前的必然过渡，语义收益低，容易放大噪声。

标注口径：人工只标 task-level primitive chain 与 object slot，用于校准自动 cluster 和审计 GT；不做全量逐帧 primitive 标注。

## 5. Dataset Roles

| Dataset | 论文角色 | 主要支持题型 | 当前状态 |
| --- | --- | --- | --- |
| `GM-100` | 主数据源；长尾、goal-conditioned、双臂任务。 | `T1-T6, T8, T9`；`T10-T12` 作为 primitive extension。 | 已落盘 `15,500` items。 |
| `REASSEMBLE` | action-chain 数据源；最干净的 primitive / outcome 外部验证。 | `T1, T2, T5-T12`；不做 `T3/T4`。 | 已落盘 `17,165` items，`37` recordings，`10` tasks。 |
| `RH20T` | force/torque + gripper + TCP 支撑的 contact/process 数据源。 | `T1, T2, T5, T6, T8, T9`；`T10/T11` pilot。 | 已落盘 `10,000` items，覆盖 `76` task IDs。 |
| `AIST-Bimanual` | 无力信号的双臂运动学数据源。 | `T4, T6, T9`；后续扩 `T3/T7`。 | 已落盘 `3,381` items，`160` episodes；下载仍在扩展。 |
| `RoboMIND2.0 Agilex` | 大规模双臂 RGB-D + proprioception 扩展源。 | `T3, T4, T6, T7, T8, T9` pilot。 | 已审计单个 `trajectory.hdf5`；未确认 force/torque 字段。 |

覆盖原则：不要求每个数据集覆盖全部 `12` 题。每个数据集只承担其信号稳定支持的题型。

## 6. Current Release Snapshot

| Source | Curated file | Frame cache | Eval status |
| --- | --- | --- | --- |
| GM-100 | `benchmark/benchmark_v1_curated.jsonl` | `benchmark/benchmark_v1_frames_tbinary_20260330/` | root benchmark 可评测。 |
| REASSEMBLE | `benchmark/reassemble_benchmark_v0_curated.jsonl` | `benchmark/reassemble_benchmark_v0_frames/` | 已接入独立 eval / score。 |
| RH20T | `benchmark/rh20t_benchmark_v0_curated.jsonl` | `benchmark/rh20t_benchmark_v0_frames/` | 已接入独立 eval / score。 |
| AIST | `benchmark/aist_benchmark_v0/aist_benchmark_v0_curated.jsonl` | 由 AIST builder 按需导出。 | GT builder 已有；eval wrapper 待并入统一入口。 |
| RoboMIND2.0 | `benchmark/robomind2_agilex_probe/` | probe frames 已导出。 | 先做 schema audit，不进入主结果承诺。 |

已落盘题量：

| Source | Items | 已覆盖题型 |
| --- | ---: | --- |
| GM-100 | `15,500` | `T1, T2, T3, T4, T5, T6, T8, T9` |
| REASSEMBLE | `17,165` | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` |
| RH20T | `10,000` | `T1, T2, T5, T6, T8, T9` |
| AIST | `3,381` | `T4, T6, T9` |

当前已落盘合计：`46,046` items。`~100k` 是全量 roll-out 目标，不写成投稿版既成结果。

## 7. GT Construction Rules

### 7.1 Shared Rules

1. 优先使用 dataset-native signal：force/torque、gripper state、TCP、joint state、velocity、official segment labels、task metadata。
2. 每个题型先定义可复现的 local unit，再采样 frame。
3. GT builder 自动生成，人工审计只验证可靠性和修正 taxonomy。
4. `task-meta` 是可控变量：主实验需报告 with-meta / without-meta 或明确默认协议。
5. `primitive-object pair` 用作辅助标注和分析，不直接升级为全题库主标签。

### 7.2 T5 / T_progress v2

`T5` 统一定义为 **within-local-step progress**。

GT 连续值：`u = (t - start(P_k)) / (end(P_k) - start(P_k))`。

发布标签：`early / middle / late`。

| Dataset | `P_k` 来源 | 备注 |
| --- | --- | --- |
| GM-100 | contact-aware local interaction interval。 | 不使用 whole-task progress。 |
| REASSEMBLE | 官方 successful high-level segment。 | 最稳定的 dataset-native progress。 |
| RH20T | force/torque + gripper + TCP 合成 local interval。 | 先以 pilot audit 校准阈值。 |
| AIST | kinematic local interval。 | 只作为后续扩展，不进入第一主表。 |
| RoboMIND2.0 | proprioception-based local interval。 | 需更多样本确认。 |

## 8. Evaluation Plan

主实验只围绕最小闭环展开。

| 实验 | 目的 | 输出 |
| --- | --- | --- |
| Cross-model benchmark | 比较 VLM 在 `12` 题上的 process-aware 能力。 | overall、per-task、per-category 表。 |
| Static vs dynamic split | 验证动态过程推理是否明显更难。 | `T1/T2/T4/T10` vs `T3/T5/T6/T7/T8/T9/T11/T12`。 |
| Source breakdown | 检查模型是否只适应单一视觉域。 | GM-100 / REASSEMBLE / RH20T / AIST 分表。 |
| With-meta analysis | 衡量 goal/task metadata 对过程理解的帮助。 | without-meta vs with-meta。 |
| Error taxonomy | 找出主要 failure modes。 | contact ambiguity、temporal confusion、primitive confusion、viewpoint bias。 |
| Base vs SFT | 验证 benchmark-derived supervision 的价值。 | 同一 open-weight VLM fine-tuning 前后对比。 |

SFT 范围控制：只训练一个 open-weight VLM。主结果比较 base vs SFT，不引入 RL 主线。

## 9. Manual Audit

人工标注员：Huiting Ji。

人工工作只保留两类：

1. `benchmark validity audit`：分层抽检题目、GT、frame 可见性和歧义。
2. `task-level primitive-object chain`：为 GM-100 / RH20T / REASSEMBLE 校准 coarse primitive chain 与 object slot。

不做三件事：

1. 不做全量逐帧 primitive 标注。
2. 不做全量 object grounding 标注。
3. 不把 semantic verb-object pair 当作所有数据集的主 GT。

审计优先级：`T5/T7/T10/T11/T12` 高于简单单帧题；RH20T 和 AIST 重点审计 viewpoint 与可见性。

## 10. Scope Control

投稿版可以主张：

1. 一个 process-aware manipulation benchmark topology。
2. 多源、信号驱动、可审计的 GT 构建方式。
3. VLM 在动态过程推理上存在系统性短板。
4. Benchmark-derived SFT 能提升过程理解能力。

投稿版避免主张：

1. 所有数据集共享精细高语义 primitive ontology。
2. 所有题型在五个数据集上完全覆盖。
3. AIST 或 RoboMIND2.0 已确认包含稳定 force/torque。
4. `~100k` questions 已全部落盘。
5. RL 是本文核心贡献。

## 11. Immediate Plan

| 优先级 | 任务 | 退出条件 |
| --- | --- | --- |
| P0 | 冻结 master plan、benchmark card、task naming。 | 文档只保留最小闭环叙事。 |
| P0 | 跑通 GM-100 / REASSEMBLE / RH20T 的统一评测表。 | 三个 source 均能 eval + score。 |
| P1 | 完成 AIST selected tasks 的 frame cache 与 eval wrapper。 | AIST 至少进入 `T4/T6/T9` 外部表。 |
| P1 | 完成 Huiting audit cards。 | 每个高风险题型有人工抽检统计。 |
| P1 | 构建 SFT train split。 | 与 eval split 严格隔离。 |
| P2 | RoboMIND2.0 扩展接入。 | 先出 schema-confirmed pilot，不影响主文提交。 |

最终论文主图建议：

1. 左：五数据源及其 signal / metadata。
2. 中：`2` 类 `12` 题 process-aware benchmark。
3. 右：cross-model evaluation 与 base-vs-SFT 闭环。
