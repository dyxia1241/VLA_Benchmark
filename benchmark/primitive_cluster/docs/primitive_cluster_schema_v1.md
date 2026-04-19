# Primitive Cluster Schema v1

更新时间：2026-04-10  
适用范围：`benchmark/primitive_cluster/`

## 1. 作用

这份 schema 定义 `primitive cluster` 工作流的最小数据模型。

覆盖对象：

- `task_semantic_canonical`
- `anchor_event`
- `primitive_cluster_proposal`
- `cluster_annotation_row`
- `adjudicated_cluster`

不覆盖：

- 实现代码
- review card 版式
- benchmark scorer 细节

## 2. 基本原则

1. raw `contact` 不是 primitive，只是 `anchor_event`
2. `anchor_event` 只保留 episode-level provenance，不直接作为 GT
3. `primitive_cluster_proposal` 是工作层，不是最终语义标注
4. `adjudicated_cluster` 才是 benchmark export 的 source of truth
5. 所有 ID 统一存成字符串
6. 标准路径统一使用 repo-relative path

## 3. 实体关系

```text
Task Semantic Canonical
        |
        v
Anchor Event
        |
        v
Primitive Cluster Proposal
        |
        v
Cluster Annotation / Adjudication
        |
        v
Adjudicated Cluster
```

约束：

1. 一个 `task_id` 对应一条 canonical chain
2. 一个 `episode_id` 对应若干个 `anchor_event`
3. 一个 `primitive_cluster_proposal` 由一个或多个相邻 `anchor_event` 组成
4. 一个 `adjudicated_cluster` 可以对应一个或多个被 merge 的 proposal

## 4. 共享字段

以下字段应尽量在核心对象间复用：

- `dataset_id`
- `task_id`
- `episode_id`
- `task_name_raw`
- `task_name_readable`
- `task_meta_text`
- `schema_version`
  - v1 固定为 `primitive_cluster_schema_v1`
- `source_version`
- `notes`

## 5. 闭集

### 5.1 `task_execution_mode`

- `single_pass`
- `serial_repetition`
- `hybrid_or_uncertain`

### 5.2 `anchor_type`

- `contact_bout`
- `other_adapter_defined`

### 5.3 `relation_to_previous_cluster`

- `first_cluster`
- `new_step`
- `retry_same_subgoal`
- `serial_repeat_same_step`
- `uncertain`

### 5.4 `cluster_validity`

- `valid`
- `invalid_noise`
- `ambiguous`
- `holdout`

### 5.5 `merge_with_previous`

- `yes`
- `no`
- `uncertain`

### 5.6 `proposal_merge_confidence`

- `high`
- `medium`
- `low`

### 5.7 `serial_repetition_risk`

- `low`
- `medium`
- `high`

## 6. `task_semantic_canonical`

建议文件：

- `benchmark/primitive_cluster/derived/task_semantic_canonical_v1.jsonl`

字段：

- `dataset_id`
- `task_id`
- `task_name_raw`
- `task_name_readable`
- `metadata_status`
- `task_meta_text`
- `canonical_chain`
  - 字符串数组；使用 compressed chain，不展开 retry 或 serial repetition
- `coordination_pattern`
- `task_execution_mode`
- `canonical_chain_source`
- `schema_version`
- `source_version`
- `notes`

约束：

1. `canonical_chain` 中的 primitive 必须来自 semantic 层闭集
2. `canonical_step_index` 一律按 1-based 引用 `canonical_chain`
3. `metadata_status != metadata_ready` 的 task 默认不进入 active cluster export

## 7. `anchor_event`

建议文件：

- `benchmark/primitive_cluster/runs/<run_tag>/anchor_events.jsonl`

字段：

- `dataset_id`
- `task_id`
- `episode_id`
- `anchor_event_id`
  - 推荐格式：`<dataset>__<task_id>__<episode_id>__a<index>`
- `anchor_order`
- `anchor_type`
- `anchor_source`
- `primary_view`
- `anchor_start_frame`
- `anchor_frame`
- `anchor_end_frame`
- `active_arm_pattern`
  - 建议闭集：`left / right / both / unknown`
- `quality_flags`
- `source_event_refs`
- `adapter_payload`
- `schema_version`
- `source_version`
- `notes`

约束：

1. `anchor_start_frame <= anchor_frame <= anchor_end_frame`
2. `anchor_order` 在同一 episode 内唯一且单调递增
3. `anchor_event` 必须是 adapter 可重复生成的稳定结果

## 8. `primitive_cluster_proposal`

建议文件：

- `benchmark/primitive_cluster/runs/<run_tag>/cluster_proposals.jsonl`

字段：

- `dataset_id`
- `task_id`
- `episode_id`
- `proposal_cluster_id`
  - 推荐格式：`<dataset>__<task_id>__<episode_id>__pc<index>`
- `proposal_order`
- `source_anchor_event_ids`
- `proposal_start_frame`
- `proposal_end_frame`
- `representative_frame_current`
- `representative_frames_context`
- `inter_anchor_gap_frames`
- `proposal_merge_confidence`
- `proposal_reason_codes`
- `serial_repetition_risk`
- `canonical_step_index_candidate`
  - 可空，仅作辅助
- `validity_flags`
- `schema_version`
- `source_version`
- `notes`

约束：

1. `source_anchor_event_ids` 必须来自同一个 episode
2. `source_anchor_event_ids` 必须是相邻 anchor 的连续子序列
3. proposal 可以偏碎，但不允许无 provenance 的新造 cluster

## 9. `cluster_annotation_row`

建议文件：

- `benchmark/primitive_cluster/runs/<run_tag>/cluster_annotation_master.csv`
- `benchmark/primitive_cluster/runs/<run_tag>/annotator_a.csv`
- `benchmark/primitive_cluster/runs/<run_tag>/annotator_b.csv`

推荐把 metadata 列和 editable 列分开。

### 9.1 metadata 列

- `dataset_id`
- `task_id`
- `episode_id`
- `proposal_cluster_id`
- `proposal_order`
- `source_anchor_event_ids`
- `proposal_start_frame`
- `proposal_end_frame`
- `representative_frame_current`
- `representative_frame_prev`
- `representative_frame_next`
- `task_meta_text`
- `canonical_chain_text`
- `task_execution_mode`
- `serial_repetition_risk`
- `proposal_merge_confidence`
- `proposal_reason_codes`
- `validity_flags`

### 9.2 editable 列

- `primitive_label`
- `canonical_step_index`
- `relation_to_previous_cluster`
- `merge_with_previous`
- `cluster_validity`
- `next_cluster_primitive`
- `next_distinct_primitive`
- `notes`

约束：

1. `cluster_validity != valid` 时，`next_*` 可以留空
2. `relation_to_previous_cluster = first_cluster` 时，`merge_with_previous` 必须为 `no`
3. `relation_to_previous_cluster = retry_same_subgoal` 时，`merge_with_previous` 默认应为 `yes`
4. `relation_to_previous_cluster = serial_repeat_same_step` 时，`merge_with_previous` 默认必须为 `no`

## 10. `adjudicated_cluster`

建议文件：

- `benchmark/primitive_cluster/runs/<run_tag>/adjudicated_clusters.jsonl`

字段：

- `dataset_id`
- `task_id`
- `episode_id`
- `final_cluster_id`
  - 推荐格式：`<dataset>__<task_id>__<episode_id>__fc<index>`
- `final_order`
- `source_proposal_cluster_ids`
- `source_anchor_event_ids`
- `cluster_start_frame`
- `cluster_end_frame`
- `representative_frame_current`
- `primitive_label`
- `canonical_step_index`
- `repeat_instance_index`
  - 同一 `canonical_step_index` 下的 1-based 重复实例编号
- `relation_to_previous_cluster`
- `next_cluster_primitive`
- `next_distinct_primitive`
- `cluster_validity`
- `eligible_current_primitive`
- `eligible_next_primitive`
- `eligible_chain_reconstruction`
- `ineligibility_reasons`
- `schema_version`
- `source_version`
- `adjudication_notes`

约束：

1. `final_order` 在 episode 内唯一且单调递增
2. `source_proposal_cluster_ids` 必须保持时间顺序
3. `primitive_label` 必须来自 semantic primitive 闭集
4. `repeat_instance_index` 只在 `cluster_validity = valid` 时使用
5. `eligible_*` 是 benchmark export gate，不等同于 `cluster_validity`

## 11. 文件约定

当前阶段固定位置：

- 根目录 README：`benchmark/primitive_cluster/README.md`
- workflow 根目录：`benchmark/primitive_cluster/`
- spec：`benchmark/primitive_cluster/docs/`

后续实现阶段推荐目录：

- `benchmark/primitive_cluster/dataset_profiles/`
- `benchmark/primitive_cluster/runs/<run_tag>/`
- `benchmark/primitive_cluster/annotator_packages/`

`benchmark/primitive_cluster/runs/<run_tag>/` 推荐至少包含：

- `run_manifest.json`
- `anchor_events.jsonl`
- `cluster_proposals.jsonl`
- `proposal_summary.json`
- `proposal_healthcheck.json`
- `review_cards/`
- `cluster_annotation_master.csv`
- `annotator_a.csv`
- `annotator_a.xlsx`
- `annotator_b.csv`
- `annotator_b.xlsx`
- `scored/`
- `adjudicated_clusters.jsonl`
- `adjudication_log.csv`

## 12. 版本规则

1. 字段增删改必须提升 schema 版本
2. v1 禁止 silent field rename
3. adapter 特定字段必须进入 `adapter_payload`
4. benchmark export 不得直接读取 annotator 原始 CSV，必须读取 `adjudicated_clusters.jsonl`
