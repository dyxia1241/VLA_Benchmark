# Dataset Adapter Contract v1

更新时间：2026-04-10  
适用范围：`benchmark/primitive_cluster/`

## 1. 作用

这份 contract 定义如何把 `primitive cluster` workflow 迁移到其他 GM-100 类数据集。

目标是把数据集特定逻辑收敛到 adapter 层，让共享 workflow 只依赖标准化输出，而不依赖原始字段名、原始目录结构或某一种接触信号实现。

## 2. 职责边界

### 2.1 共享层负责

- 消费标准化 `task_semantic_canonical`
- 消费标准化 `anchor_event`
- 执行统一 merge policy
- 导出统一 annotation package
- 汇总 annotator disagreement
- 读取 `adjudicated_cluster` 导出 benchmark item

### 2.2 Adapter 负责

- 枚举 task 与 episode
- 生成统一 task metadata
- 提供 frame / image 解析方式
- 从原始信号抽取 `anchor_event`
- 为相邻 anchor 计算 merge features
- 提供 holdout / eligibility 提示

## 3. Dataset Profile

每个数据集应提供一份 profile，建议位置：

- `benchmark/primitive_cluster/dataset_profiles/<dataset_id>_profile_v1.yaml`

推荐字段：

- `dataset_id`
- `dataset_display_name`
- `dataset_root_rel`
- `task_table_source_rel`
- `episode_inventory_source_rel`
- `default_view`
- `task_meta_source`
- `frame_access_mode`
- `anchor_event_source_type`
- `arm_taxonomy_mapping`
- `task_execution_mode_source`
- `eligible_task_filters`
- `holdout_rules`
- `adapter_version`

这份 profile 只存配置，不承载大体量数据。

## 4. 标准化表

### 4.1 Task Table

至少包含：

- `dataset_id`
- `task_id`
- `task_name_raw`
- `task_name_readable`
- `metadata_status`
- `task_meta_text`
- `arm_type`
- `primary_arm`
- `default_view`
- `task_execution_mode_candidate`
- `source_task_meta_ref`

### 4.2 Episode Inventory

至少包含：

- `dataset_id`
- `task_id`
- `episode_id`
- `episode_rel_path`
- `default_view`
- `frame_index_min`
- `frame_index_max`
- `n_frames`
- `episode_quality_flags`

## 5. Adapter 必须提供的能力

逻辑上至少支持：

1. `enumerate_tasks()`
2. `enumerate_episodes(task_id)`
3. `resolve_task_meta(task_id)`
4. `resolve_frame(task_id, episode_id, frame_index, view)`
5. `extract_anchor_events(task_id, episode_id)`
6. `compute_pairwise_merge_features(anchor_i, anchor_j)`
7. `infer_holdout_flags(task_id, episode_id)`

不强制实现语言，也不强制具体函数签名。

## 6. `anchor_event` 适配要求

新数据集不一定有 `contact`，但必须能提供某种可重复抽取的 `anchor_event`。

可接受的来源例如：

- effort / force / contact detection
- gripper state transition
- tool-use activation event
- externally provided event marker

要求：

1. 同一 adapter 版本下可重复生成
2. 能定位到 episode 内时间顺序
3. 能映射到 review card 所需 frame
4. 能为相邻 pair 计算 merge features

## 7. Pairwise Merge Feature Contract

每对相邻 anchor 至少输出：

- `gap_frames`
- `same_active_arm_pattern`
- `inter_anchor_reset_hint`
- `explicit_retry_hint`
- `serial_repetition_hint`
- `inter_anchor_transition_strength`
- `view_stability_hint`

adapter 可以额外输出更多 dataset-specific 信息，但必须放到：

- `adapter_payload`
- `adapter_reason_codes`

共享 merge engine 不得把这些扩展字段作为硬依赖。

## 8. Frame / Asset 解析要求

adapter 需要明确：

- 默认使用哪个 view
- frame 按 index 还是 timestamp 寻址
- 是否能稳定导出 repo-relative path
- 是否需要先 materialize preview assets

推荐约定：

1. review card 使用 repo-relative path 或可复现 frame key
2. adapter 不把 GUI / notebook 专用路径写进标准 JSONL
3. 标准输出只暴露 benchmark 需要的最少视觉证据

## 9. Holdout 与 Eligibility

adapter 必须提供 task / episode 层 gate 提示，不能把所有数据默认当作 active candidate。

至少支持以下 gate：

- `missing_metadata`
- `missing_view_or_frame`
- `unstable_anchor_signal`
- `adapter_defined_holdout`

这些 gate 只做前置过滤，不替代后续人工裁决。

## 10. GM-100 v1 推荐映射

- `dataset_id`: `gm100`
- `task table`: `gt_build/task_type_annotation.csv` + dataset `tasks.jsonl` + `GM100 List.xlsx`
- `default_view`: `camera_top`
- `task_meta_text`: 优先 `GM100 List.xlsx`，缺失时 fallback 到 dataset task name
- `anchor_event_source_type`: `effector_effort_contact_v1`
- `anchor_type`: `contact_bout`
- `arm taxonomy`: `single_left / single_right / bimanual_sync / bimanual_sequential`

GM-100 是第一套 adapter，不应把 shared workflow 命名成 GM-100 专用实现。

## 11. 新数据集接入 Checklist

复用这套 workflow 前，至少先回答：

1. 是否有稳定的 `task_id` 与 `episode_id`？
2. 是否能稳定取到默认视角 frame？
3. 是否能定义一种可重复抽取的 `anchor_event`？
4. 是否能为相邻 anchor 提供 merge feature？
5. 是否能给 task 提供可读的 task meta？
6. 是否能识别 `single_pass / serial_repetition / hybrid_or_uncertain` 候选？
7. 是否能定义 holdout 规则？

若以上任一问题回答为否，则不建议直接复用当前 workflow。

## 12. v1 边界

v1 不要求：

1. object-slot binding 已完整解决
2. dense timeline 已存在
3. 现成 primitive GT 已存在
4. 所有任务都适合 contact-anchored cluster

v1 只要求数据集能提供一条稳定的：

- `anchor_event -> proposal -> adjudication`

这样 shared workflow 才能同时在 GM-100 和其他类似数据集上落地。
