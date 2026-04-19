# Primitive Cluster

更新时间：2026-04-10  
适用范围：`benchmark/primitive_cluster/`

## 1. 作用

本目录是 `primitive cluster` 这条线的唯一根目录。

这里统一承载：

1. 战略目标
2. workflow spec
3. pilot 脚本
4. 运行产物

这条线位于：

- `task-level semantic` 之后
- dense `segment trace` 之前

旧的 `benchmark/PRIMITIVE_CLUSTER_STRATEGY.md` 已并入本文件，不再单独维护。

## 2. 目标

目标不是做 dense primitive timeline，而是建立一套更轻量、可执行、可迁移的 benchmark-side workflow：

1. 先定义 task-level `canonical primitive chain`
2. 再自动生成 episode-level `primitive cluster proposal`
3. 再由人工把 `cluster` 对齐到 `primitive`
4. 最后导出新的 benchmark 题目

## 3. 核心判断

1. raw `contact` 不是 primitive，只是 `anchor event`
2. 真正适合标注和出题的单位是 `primitive cluster`
3. `retry_same_subgoal` 通常应 merge 到同一个 cluster
4. `serial_repeat_same_step` 通常不应 merge
5. v1 不做 dense per-frame primitive timeline

## 4. 四步工作流

### Step 1. Task-Level Canonical Chain

人工观看 task 视频与 meta，只给每个 task 一个按发生顺序排列的 `canonical primitive chain`。

这一层：

- 不对齐时间轴
- 不展开 retry
- 不展开 serial repetition

产物：

- `task_semantic_canonical`

### Step 2. Episode-Level Cluster Proposal

系统基于每个 episode 的 `anchor event` 自动生成 `primitive cluster proposal`。

这一层：

- 从 `contact / hold / 其他可重复抽取的 anchor` 出发
- 自动 merge 明显属于同一子目标的重复接触或微调
- 先得到 episode 里有哪些 cluster
- 暂时不要求 cluster 已有最终 `primitive_label`

产物：

- `anchor_event`
- `primitive_cluster_proposal`

### Step 3. Cluster Annotation / Adjudication

人工对 proposal cluster 做语义裁决，而不是回到逐帧时间轴标注。

这一层要回答：

1. 当前 cluster 对应哪个 `primitive`
2. 它和前一个 cluster 是 `new_step`、`retry_same_subgoal` 还是 `serial_repeat_same_step`
3. 是否需要继续 merge
4. 它之后的 `next primitive` 是什么

这一步完成后，得到的是：

- 每个 episode 的 `cluster` 序列
- 每个 cluster 的时间范围
- 每个 cluster 的 `primitive_label`

这是稀疏但结构化的时间对齐，不是 dense timeline。

产物：

- `cluster_annotation_row`
- `adjudicated_cluster`

### Step 4. Benchmark Export

系统从最终的 `adjudicated_cluster` 导出 benchmark item。

当前优先支持：

1. `current primitive`
2. `next primitive`
3. `primitive chain reconstruction`

其中：

- `current primitive` 和 `next primitive` 是主线候选题
- `primitive chain reconstruction` 先作为 sidecar，不默认并入主总分

## 5. 目录结构

当前固定结构：


- `README.md`
  - 战略总纲与目录入口
- `docs/`
  - `primitive_cluster_schema_v1.md`
  - `merge_policy_v1.md`
  - `annotation_guideline_v1.md`
  - `dataset_adapter_contract_v1.md`
- `build_cluster_pilot.py`
  - 每个 task 先跑一个代表 episode，估计 cluster 数
- `smoke_test_cluster_consistency.py`
  - 抽少量 episode，检查 task 内 `cluster_count` 是否稳定
- `runs/`
  - pilot 与 smoke test 产物

规划中的后续子目录：

- `derived/`
- `dataset_profiles/`
- `annotator_packages/`

## 6. 当前实现状态

当前已经有两类 pilot：

1. `one_episode_per_task_v0`
   - 每个 task 选一个代表 episode，先看大概有几个 cluster
2. `smoke_consistency_v0`
   - 对 sampled task 再抽少量 episode，看 task 内 `cluster_count` 是否稳定

当前实现是 `v0 heuristic proposal`：

- 目标是先估计 cluster 数量
- 不是最终 GT
- merge 逻辑刻意保守，宁可略碎，不主动过度合并

## 7. 当前边界

1. v1 不做 dense primitive timeline
2. 当前 `0 cluster` 更准确地说是 `detector_zero`，不等于语义上没有 interaction
3. 先做 contact-rich subset，不追求一次覆盖全部 task
4. 当前 task 内 `cluster_count` 不能直接假设全 episode 完全一致
5. 这条线本质上是 benchmark-side structured process reasoning workflow
