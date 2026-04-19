# Primitive Cluster Annotation Guideline v1

更新时间：2026-04-10  
适用范围：`benchmark/primitive_cluster/`

## 1. 标注目标

你现在标注的不是 dense primitive timeline，而是 `primitive cluster proposal`。

v1 的核心任务只有四件：

1. 给当前 cluster 标 `primitive_label`
2. 判断它与前一个 cluster 的关系
3. 判断它是否应 merge 到前一个 cluster
4. 标它之后的 `next primitive`

## 2. 当前不要求做的事

v1 不要求 annotator：

1. 逐帧画完整 primitive timeline
2. 给整个 episode 做 dense segmentation
3. 靠自由文本重写整条任务脚本
4. 自行扩充 primitive 闭集

## 3. 标注包应包含什么

每个 proposal cluster 的 review package 至少应包含：

- 当前 cluster 的代表帧
- 前一个 cluster 的参考帧
- 后一个 cluster 的参考帧
- 当前 proposal 的时间范围
- 当前 proposal 的 source anchor 数量
- task meta 文本
- task 的 canonical primitive chain
- proposal reason codes
- `serial_repetition_risk`

默认工作单元是“一行一个 proposal cluster”，不是“一行一个 frame”。

## 4. Editable 字段

需要填写：

- `primitive_label`
- `canonical_step_index`
- `relation_to_previous_cluster`
- `merge_with_previous`
- `cluster_validity`
- `next_cluster_primitive`
- `next_distinct_primitive`
- `notes`

## 5. 推荐标注顺序

按以下顺序做：

1. `cluster_validity`
2. `primitive_label`
3. `canonical_step_index`
4. `relation_to_previous_cluster`
5. `merge_with_previous`
6. `next_cluster_primitive`
7. `next_distinct_primitive`

原因很简单：当前 cluster 自身如果不稳定，后续字段也不会稳定。

## 6. 字段规则

### 6.1 `primitive_label`

1. 一个 cluster 只填一个主 primitive
2. 优先使用 task semantic 已冻结的 primitive 闭集
3. 不要把 retry、对象数量、失败状态编码进 primitive 名本身

### 6.2 `canonical_step_index`

1. 使用 1-based 编号，指向 compressed canonical chain
2. 多个 `serial_repeat_same_step` cluster 可以共享同一个索引
3. `retry_same_subgoal` 若最终并入同一步，通常共享同一个索引

### 6.3 `relation_to_previous_cluster`

闭集：

- `first_cluster`
- `new_step`
- `retry_same_subgoal`
- `serial_repeat_same_step`
- `uncertain`

判断规则：

- `new_step`
  - 已进入 canonical chain 的下一个 distinct step
- `retry_same_subgoal`
  - 仍在做同一子目标，只是失败重试或微调再试
- `serial_repeat_same_step`
  - primitive 相同，但已经在处理新的对象实例或新的重复轮次

### 6.4 `merge_with_previous`

闭集：`yes / no / uncertain`

默认规则：

- `first_cluster` -> `no`
- `new_step` -> `no`
- `retry_same_subgoal` -> 默认 `yes`
- `serial_repeat_same_step` -> 默认 `no`

### 6.5 `cluster_validity`

闭集：

- `valid`
- `invalid_noise`
- `ambiguous`
- `holdout`

解释：

- `valid`
  - 作为最终 cluster 保留
- `invalid_noise`
  - proposal 没有稳定语义，不保留
- `ambiguous`
  - 证据不足，暂时不能稳定裁决
- `holdout`
  - 当前 task / episode 不进入 active export

### 6.6 `next_cluster_primitive`

指紧邻的下一个有效 cluster 的 primitive。

规则：

1. 即使下一个 primitive 与当前相同，也照实填写
2. 若当前已是最后一个有效 cluster，则填 `terminal`
3. 若后续都无效或不确定，可留空并在 `notes` 说明

### 6.7 `next_distinct_primitive`

指后续第一个与当前不同的有效 primitive。

规则：

1. 若下一个 cluster primitive 相同，则继续向后找
2. 若后续再无不同 primitive，则填 `terminal`
3. 该字段是 `next primitive` 题的重要候选 GT

## 7. 如何区分三种关系

### 7.1 `retry_same_subgoal`

优先判 retry 的情况：

1. 当前 cluster 与前一个 cluster 在做同一个局部目标
2. 当前更像前一个失败后的再试一次
3. 中间没有进入新的对象实例或新的计数轮次
4. merge 后不会改变 canonical chain 的 distinct step 数

### 7.2 `serial_repeat_same_step`

优先判 serial repetition 的情况：

1. `primitive_label` 相同，但对象实例变了
2. 是 one-by-one / repeated placing / repeated sorting / repeated loading 一类模式
3. 当前 cluster 应被视作同一步语义在新的轮次上再次发生
4. merge 会抹掉 episode-level 计数结构

### 7.3 `new_step`

优先判 new step 的情况：

1. 当前 cluster 已进入 canonical chain 的下一步
2. semantic goal 已变化
3. active interface 或 active object 的功能角色明显变化
4. 当前不是重复前一步，而是在继续往下执行

## 8. Canonical Chain 的使用方式

annotator 参考的是 compressed canonical chain，而不是 episode-expanded chain。

这意味着：

1. canonical chain 只表达 distinct primitive step
2. retry 不在 canonical chain 中额外加步
3. serial repetition 也不在 canonical chain 中展开成多次重复
4. episode-level 的重复结构靠多个 cluster 和 `repeat_instance_index` 表达

## 9. Holdout

以下情况默认不进入 active annotation：

1. `metadata_status != metadata_ready`
2. canonical chain 尚未裁决完成
3. view / frame 证据明显不足
4. adapter 明确标记当前 episode 不适合 export

若某行进入表后仍判断不应继续，填写：

- `cluster_validity = holdout`
- 其他结构字段保持空白或最小必要填写

## 10. Annotator 工作规则

1. A/B annotator 必须独立标注，不讨论具体样本
2. 不要改 metadata 列、排序和文件名
3. 不要自行引入新的 primitive 名称
4. 不要试图靠猜测还原完整时间轴
5. proposal 看起来过碎时，优先用 `merge_with_previous` 处理，不要只写 `notes`

## 11. Adjudication 优先级

冲突处理建议顺序：

1. `cluster_validity`
2. `primitive_label`
3. `relation_to_previous_cluster`
4. `merge_with_previous`
5. `canonical_step_index`
6. `next_cluster_primitive`
7. `next_distinct_primitive`

必须保留：

- 原始 annotator CSV
- merged disagreement 表
- adjudication log

## 12. v1 的判断标准

v1 不是追求“把每个局部动作都切出来”，而是追求：

1. cluster 是否足够稳定
2. `primitive_label` 是否足够一致
3. retry 与 serial repetition 是否能被稳定区分
4. `current primitive` 与 `next primitive` 是否能可靠导出
