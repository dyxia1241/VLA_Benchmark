# GM-100 Benchmark

GM-100 的多模态评测仓库，聚焦机械臂任务上的数据构建、抽帧评测与结果分析。

当前主线已经收敛到：

- `benchmark/`：benchmark_v1 的 GT 构建、采样、抽帧、评测与评分。
- `benchmark/manual_audit/gt_audit/`：item-level 人工审计。
- `benchmark/manual_audit/semantic_affordance_audit/`：task-level semantic / affordance 标注。

## 目录

- `benchmark/`：主工作区（GT 构建、采样、抽帧、评测、评分）。
- `gm100-cobotmagic-lerobot/`：原始数据集（本地使用，不建议入库）。
- `GM100_bimanual_fullscan_20260318/`：早期 fullscan / 分型溯源产物。
- `benchmark/gt_build/task_type_annotation.csv`：当前 GT 构建与 semantic metadata 使用的 task 注释表。

## 快速开始

```bash
cd benchmark
./run_v1_pipeline.sh --help
```

## 说明

大体量产物（抽帧、评测结果、full GT）默认不建议提交到 Git。
更具体的当前状态、默认入口和手工审计说明见 [benchmark/README.md](benchmark/README.md)。
