# ProcessBench：机器人操作过程理解基准

**ProcessBench** 是一个面向**机器人操作过程理解**（robotic manipulation process understanding）的多源评测基准。  
它关注的不是静态物体识别，也不是单一接触检测，而是模型能否真正理解一段机器人操作在**当前处于什么阶段、接下来会发生什么、局部动作推进到哪里、最终是否会成功**。

当前稳定主线已经收敛为一个 **goal-conditioned、signal-grounded、process-aware** 的 manipulation benchmark，覆盖四个异构数据源、十二类任务，以及严格 episode-level 隔离的 `SFT / Eval` protocol。

## 1. 项目定位

本仓库当前服务于一条明确的论文主线：

> ProcessBench: A goal-conditioned, signal-grounded, process-aware benchmark for robotic manipulation.

核心问题不是“看到了什么物体”，而是“理解这段操作过程本身”。具体包括：

- 当前处于哪个过程阶段
- 是否已经建立有效接触
- 主要运动方向是什么
- 当前局部操作处于 early / middle / late 哪一段
- 这次尝试最终会成功还是失败
- 当前低层动作是什么、下一个动作应该是什么
- 局部动作链能否恢复

## 2. Benchmark 概览

当前主线 benchmark 覆盖四个数据源：

- `GM-100`
- `RH20T`
- `REASSEMBLE`
- `AIST`

当前冻结规模：

- 全量题库：`57,892` items
- Eval split：`9,051` items
- SFT split：`48,841` items

当前默认 split：

- `85%` SFT
- `15%` Eval
- 严格 `episode / recording / scene` 级隔离

## 3. 任务设计

十二个任务分成两大类：

### Static Process Monitoring

- `T1` Phase Recognition
- `T2` Contact Detection
- `T4` Bimanual Coordination State
- `T10` Current Primitive Recognition

### Dynamic Process Reasoning

- `T3` Motion Direction Prediction
- `T5` Primitive-local Progress
- `T6` Motion State Recognition
- `T7` Operation Outcome Prediction
- `T8` Temporal Ordering
- `T9` Temporal Priority Prediction
- `T11` Next Primitive Prediction
- `T12` Primitive Chain Restoration

命名约定：

- `T5 == T_progress`
- `T8 == T_temporal`
- `T9 == T_binary`

论文统一使用 `T5 / T8 / T9`，工程代码保留兼容旧名。

## 4. 数据源角色

四个数据源不是平均覆盖全部任务，而是各自承担其最稳定支持的题型。

| 数据源 | 角色 | 当前支持任务 | 当前规模 |
| --- | --- | --- | ---: |
| `GM-100` | 主数据源；goal-conditioned；覆盖单臂与双臂过程理解 | `T1, T2, T3, T4, T5, T6, T8, T9` | `15,500` |
| `RH20T` | force/torque + gripper + TCP 驱动的 process/contact 数据源 | `T1, T2, T3, T5, T6, T7, T8, T9` | `15,800` |
| `REASSEMBLE` | 最干净的 action-chain / primitive / outcome 数据源 | `T1, T2, T5, T6, T7, T8, T9, T10, T11, T12` | `17,165` |
| `AIST` | 无力信号的双臂运动学数据源 | `T3, T4, T6, T8, T9` | `9,427` |

## 5. 当前默认运行主线

1. 四个数据集保持各自 eval jsonl
2. 用统一总控脚本顺序运行四个 eval set
3. 再统一汇总结果

默认入口：

```bash
cd /data/projects/GM-100

python3 benchmark/eval_v1/run_all_eval_sets.py \
  --model <model> \
  --api-key <key>

python3 benchmark/eval_v1/score_all_eval_sets.py \
  --model <model>
```

当前主线最小 runtime 闭环依赖：

- `benchmark/eval_v1/run_all_eval_sets.py`
- `benchmark/eval_v1/run_pilot_eval.py`
- `benchmark/eval_v1/score_all_eval_sets.py`
- `benchmark/splits_v1/eval_manifest.json`
- 四个 per-source eval jsonl
- 四个 `*_frames/` 目录

## 6. 仓库结构

```text
GM-100/
├── raw_data/                         # 原始数据目录
│   ├── gm100-cobotmagic-lerobot/
│   ├── reassemble-tuwien-researchdata/
│   └── aist-bimanip/
├── benchmark/
│   ├── ARCHITECTURE_MAP.md          # 当前稳定主线、最小运行闭环、默认入口
│   ├── NEURIPS_ED_MASTER_PLAN.md    # 论文主线、任务定义、投稿边界
│   ├── benchmark_card.md            # benchmark 对外说明
│   ├── eval_v1/                     # 评测执行与汇总入口
│   ├── gt_build/                    # GT 构建、抽帧、split 构建
│   ├── splits_v1/                   # 当前默认 SFT / Eval split
│   └── eval_results_v1/             # 结果输出目录
└── README.md
```

## 7. 推荐阅读顺序

如果你第一次进入这个仓库，建议按下面顺序读：

1. `README.md`：快速理解 ProcessBench 在做什么
2. `benchmark/NEURIPS_ED_MASTER_PLAN.md`：理解论文主线和 benchmark 定位
3. `benchmark/ARCHITECTURE_MAP.md`：理解当前主线的目录结构与运行闭环
4. `benchmark/benchmark_card.md`：理解 benchmark 定义、协议与 caveats

## 8. 当前状态与边界

当前投稿最小闭环已经收敛为四件事：

1. 定义 process-aware benchmark topology
2. 构建四源 benchmark 题库
3. 构建严格 `episode-level` 隔离的 `SFT / Eval` split
4. 提供一键运行全部 eval set 的执行与汇总入口

## 9. 说明

- 当前主线以 `benchmark/` 目录为核心工作区。
- `benchmark/run_v1_pipeline.sh` 只保留为旧的 `GM-100` 单源兼容链路，不是当前四源主线。
- 当前主线运行逻辑、目录职责和依赖边界，以 `benchmark/ARCHITECTURE_MAP.md` 为准。
