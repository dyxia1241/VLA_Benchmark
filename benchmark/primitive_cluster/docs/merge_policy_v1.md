# Primitive Cluster Merge Policy v1

更新时间：2026-04-10  
适用范围：`benchmark/primitive_cluster/`

## 1. 作用

这份 policy 定义如何从 `anchor_event` 生成 `primitive_cluster_proposal`。

proposal 阶段的目标不是直接得到 GT，而是得到一份：

1. provenance 完整
2. 默认偏保守
3. 便于 annotator 继续 merge / relabel
4. 可迁移到其他数据集

## 2. 基本原则

1. proposal 阶段宁可略碎，不可过度合并
2. 自动 merge 只优先处理 retry，不主动吞并 serial repetition
3. 每次 merge 都必须保留 `source_anchor_event_ids`
4. proposal 不负责最终 `primitive_label`
5. `canonical_step_index_candidate` 只作辅助，不作 GT

## 3. 输入与决策顺序

proposal 阶段按以下顺序执行：

1. 枚举同一 episode 内按时间排序的 `anchor_event`
2. 只比较相邻 anchor，不允许跳跃比较
3. 先应用 `hard no-merge guards`
4. 再检查 `positive merge criteria`
5. 输出 `proposal_merge_confidence`、`proposal_reason_codes` 和 `serial_repetition_risk`

## 4. Adapter 必须提供的 pairwise features

shared merge logic 只消费标准化特征，不直接读取数据集原始信号。

每对相邻 anchor 至少提供：

- `gap_frames`
- `same_active_arm_pattern`
  - `yes / no / unknown`
- `inter_anchor_reset_hint`
  - `yes / no / unknown`
- `explicit_retry_hint`
  - `yes / no / unknown`
- `serial_repetition_hint`
  - `yes / no / unknown`
- `inter_anchor_transition_strength`
  - `low / medium / high`
- `view_stability_hint`
  - `stable / unstable / unknown`

## 5. Gap Bucket

`gap_frames` 统一分四档：

- `immediate`: `0-15`
- `short`: `16-45`
- `medium`: `46-90`
- `long`: `>90`

v1 默认只允许在 `immediate` 或 `short` 档发生自动 merge。

## 6. Hard No-Merge Guards

命中以下任一条件时，当前 pair 必须保持分裂：

1. `gap_bucket = long`
2. `serial_repetition_hint = yes`
3. `inter_anchor_reset_hint = yes`
4. `same_active_arm_pattern = no`
5. `view_stability_hint = unstable`
6. 上游 `task_execution_mode = serial_repetition` 且 `explicit_retry_hint != yes`
7. 任一 anchor 有严重质量风险

命中 hard guard 时，必须写入阻断类 `proposal_reason_codes`。

## 7. Positive Merge Criteria

### 7.1 高置信 merge

满足以下全部条件时自动 merge：

1. `gap_bucket` 属于 `immediate` 或 `short`
2. `explicit_retry_hint = yes`
3. `same_active_arm_pattern != no`
4. `inter_anchor_reset_hint = no`
5. `serial_repetition_hint != yes`
6. merge 后的 `source_anchor_event_ids` 数量不超过 `3`

输出：

- `proposal_merge_confidence = high`

### 7.2 中置信 merge

满足以下全部条件时自动 merge：

1. `gap_bucket = immediate`
2. `explicit_retry_hint != no`
3. `same_active_arm_pattern = yes`
4. `inter_anchor_reset_hint = no`
5. `serial_repetition_hint = no`
6. `inter_anchor_transition_strength = low`
7. merge 后的 `source_anchor_event_ids` 数量不超过 `2`

输出：

- `proposal_merge_confidence = medium`

### 7.3 低置信情况

以下情况不自动 merge，只保留分裂并打风险标记：

- `gap_bucket = medium`
- `same_active_arm_pattern = unknown`
- `explicit_retry_hint = unknown`
- `serial_repetition_hint = unknown`
- `inter_anchor_transition_strength = medium`

输出：

- `proposal_merge_confidence = low`
- `proposal_reason_codes` 写明原因

## 8. Proposal Reason Codes

v1 冻结以下闭集。

正向：

- `adjacent_immediate_gap`
- `adjacent_short_gap`
- `same_arm_pattern`
- `explicit_retry_hint`
- `low_transition_strength`
- `no_reset_evidence`

阻断：

- `long_gap_block`
- `serial_repetition_risk_block`
- `reset_hint_block`
- `arm_pattern_mismatch_block`
- `unstable_view_block`
- `quality_issue_block`
- `uncertain_pair_keep_split`

## 9. Retry 与 Serial Repetition 的默认处理

1. `retry_same_subgoal` 是 proposal 阶段唯一优先自动 merge 的重复类型
2. `serial_repeat_same_step` 默认不自动 merge
3. 即使两个 cluster 未来可能有相同 `primitive_label`，proposal 期也先保留分裂
4. proposal 阶段禁止跨越中间 cluster 做跳跃式 merge
5. proposal 阶段不试图判断“整个 episode 是否属于同一个长 primitive”

## 10. GM-100 v1 默认解释

1. `anchor_event` 默认来自一次 `contact-release` cycle
2. `explicit_retry_hint` 优先由短 gap、局部重复接触、无明显 reset 触发
3. `serial_repetition_hint` 优先由 `task_execution_mode` 和 task meta 中的 one-by-one / multiple-item / repeated placing 模式触发
4. 当前 proposal 期不要求 object identity 或 slot binding 已解决

## 11. Output 约束

每个 `primitive_cluster_proposal` 必须满足：

1. `source_anchor_event_ids` 连续且可追溯
2. `proposal_start_frame` 来自首个 source anchor
3. `proposal_end_frame` 来自末个 source anchor
4. `proposal_reason_codes` 足以解释为何 merge 或为何保持分裂
5. `serial_repetition_risk` 不允许留空

## 12. 需要人工重点回看的情况

以下情况应优先进入 annotation 阶段的重点复核：

1. `proposal_merge_confidence = low`
2. `serial_repetition_risk = high`
3. 同一 episode 内 proposal cluster 数异常高
4. `proposal_reason_codes` 同时出现正向与阻断信号
5. scorer 发现 annotator 对 merge 冲突高度集中

## 13. v1 不做的事

1. 不用 proposal 直接输出最终 `primitive_label`
2. 不在 proposal 阶段做 dense timeline
3. 不在 proposal 阶段展开 full episode primitive chain
4. 不把 object-slot binding 作为 proposal 前置条件
