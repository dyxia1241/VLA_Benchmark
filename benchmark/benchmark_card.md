---
pretty_name: "BiManip-Bench"
license: "mit"
tags:
  - robotics
  - benchmark
  - vision-language-models
  - manipulation
  - bimanual
  - video
  - image
  - mlcroissant
task_categories:
  - image-classification
  - visual-question-answering
task_ids:
  - visual-question-answering
language:
  - en
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: test
        path: "benchmark_v1_curated.jsonl"
---

# BiManip-Bench: Benchmark Card (v1.1 Draft Aligned to Current Root Release)

## 1. Overview

**Name:** BiManip-Bench  
**Version:** v1.1-draft (current root working release)  
**Release date:** 2026-03-31 (current root release; public release date TBD)  
**Primary contribution type:** benchmark suite + evaluation protocol  
**Primary venue target:** NeurIPS 2026 Datasets & Benchmarks Track  
**Secondary venue target:** CoRL 2026  
**Source dataset:** GM-100 (Cobot Magic subset, Lerobot format)  
**Robot platform:** Agilex Cobot Magic  
**Primary view in v1:** `camera_top`  
**Codebase root:** `/data/projects/GM-100`  
**Benchmark root:** `/data/projects/GM-100/benchmark`  
**Supported artifact URL(s):**
- Code: TBD (repository URL to be filled at public release)
- Data: TBD (HF dataset URL to be filled at public release)
- Paper/project page: optional (TBD)

## 2. Motivation

BiManip-Bench is designed to diagnose whether a VLM can perceive manipulation-relevant visual signals in bimanual teleoperation trajectories, rather than only score well on generic multimodal benchmarks.

**The gap this benchmark addresses**: recent VLA-oriented studies increasingly point to visual perception as a major bottleneck for manipulation performance, but existing evaluation sets still under-specify which visual dimensions are failing. In particular, many prior robotics-VLM evaluations are built around single-arm settings, which makes it difficult or impossible to construct tasks that require simultaneous observation of both arms, role asymmetry, or coordination state.

**Why GM-100 and why bimanual teleoperation**: the GM-100 Cobot Magic subset provides synchronized robot-side signals such as effector effort and arm velocity, allowing automatic GT construction from physical signals without full manual per-frame labeling. This makes it feasible to probe contact, phase, motion state, and coordination with consistent labeling logic at scale.

**Why `camera_top` in v1**: the top view provides the most stable simultaneous visibility of both manipulators across tasks. That design choice is especially important for coordination-sensitive tasks such as `T4`, and it reduces view-induced variance during the first benchmark release.

v1 therefore emphasizes:
- contact-sensitive visual states
- dual-arm coordination patterns
- coarse manipulation phase understanding
- motion-state discrimination
- pairwise temporal discrimination
- coarse progress estimation from manipulation state
- protocol-stable temporal ordering evaluation

## 3. Evaluative Role

### 3.1 Primary evaluative claim
BiManip-Bench supports the following specific claim: *VLMs differ systematically in their ability to perceive manipulation-relevant visual signals in bimanual robotic trajectories, and these differences can be measured through structured probing tasks with physics-derived GT under a fixed top-view protocol.*

### 3.2 What the benchmark is intended to measure
- coarse phase/state recognition (`T1`)
- contact perception around gripper-object interaction (`T2`)
- dominant motion direction perception (`T3`)
- bimanual activity-state recognition (`T4`)
- motion-state recognition (`T6`, binary in current v1)
- pairwise temporal discrimination (`T_binary`, current root release, protocol caveat)
- trajectory progress understanding (`T_progress`)
- temporal ordering from shuffled frame triples (`T_temporal`)

### 3.3 What the benchmark does NOT claim
- downstream VLA policy quality guarantees
- deployment success prediction without control-loop experiments
- cross-robot or cross-lab transfer guarantees
- generic all-purpose VLM ranking validity

### 3.4 Score interpretation
- read per-task scores first, then overall score
- always compare against random baseline
- report arm-type and task-family breakdowns
- treat `T_temporal` as protocol-sensitive diagnostic in v1, not headline metric

## 4. Intended Use

- diagnostic evaluation of VLM perception for manipulation
- model comparison under robotics-relevant visual probes
- failure mode analysis by task family / arm type
- ablation on protocol design, parser robustness, and sampling strategy

## 5. Out-of-Scope Use

- claiming causal VLA control improvement from benchmark score alone
- ranking general-purpose VLM capability as a whole
- claiming robustness on unseen robots/views/labs without dedicated tests
- safety/deployment certification decisions

## 6. Dataset Provenance

### 6.1 Source dataset
Derived from public GM-100 trajectories in Lerobot format, using locally available Cobot Magic subset.

### 6.2 Collection setting
- collection mode: human teleoperation
- environment: controlled lab setup
- trajectory type: successful demonstrations
- modality used in v1 benchmark items: top-view RGB frames
- robot-side signals used for auto-GT: effector effort, arm velocity, action arm position delta, frame index, task metadata

### 6.3 Why top view in v1
Top view gives stable dual-arm visibility and consistent processing. Multi-view sanity checks exist in earlier exploratory work, but are not part of v1 benchmark definition.

## 7. Benchmark Composition

### 7.1 Unit of evaluation
Each item contains:
- one or more extracted frames
- natural-language question
- multiple-choice options or temporal-label protocol fields
- ground-truth answer
- metadata (task id, episode id, task family, and optional arm info)

### 7.2 Task inventory (current implemented v1)

| Task ID | Name | Capability | Input form at eval | Label source | Random baseline | Role in current v1 |
|---|---|---|---|---|---|---|
| `T1` | Phase Recognition | coarse phase state | single frame | auto segmentation (`effort+velocity`) | 25% | main |
| `T2` | Contact Detection | contact perception | single frame | `effector.effort` contact-event logic | 50% | weak-signal |
| `T3` | Motion Direction | dominant direction cue | 4-frame ordered context (`-6,-3,0,+3`) | `action.arm.position` delta | 25% | hard |
| `T4` | Bimanual Coordination | dual-arm activity state | single frame | left/right velocity activity | 25% | main |
| `T6` | Motion State (binary) | moving vs stationary | 5-frame ordered context (`t-6..t+6`) | sequence-mean velocity thresholding | 50% | main (bias caveat) |
| `T_temporal` | Temporal Ordering | sequence ordering | 3 shuffled frames + `X/Y/Z` labels | frame index ordering | 16.7% | protocol-sensitive |
| `T_binary` | Binary Frame Ordering | pairwise temporal discrimination | single composite image with two labeled panels (`X`,`Y`) | frame index comparison | 50% | experimental diagnostic |
| `T_progress` | Task Progress | coarse completion stage | single frame | stage-to-progress mapping | 20% | weak-signal |

### 7.3 Scale (current root release)

**Current root release** (`benchmark_v1_curated.jsonl`, 8-family):

- total curated items: `15,500`
- source task coverage: `106` tasks
- task-family distribution:
  - `T1=3000`
  - `T2=1000`
  - `T3=2500`
  - `T4=1500`
  - `T6=1500`
  - `T_temporal=2000`
  - `T_binary=1500`
  - `T_progress=2500`
- extracted frame cache (current main cache): `benchmark_v1_frames_tbinary_20260330`
  - jpg files: `34,312`
  - extraction failures: `0`
- `T_binary v2` sampled answer distribution: `X=738`, `Y=762`
- `T_binary v2` difficulty distribution (sampled): `easy_cross_stage=895`, `hard_adjacent_stage=605`
- `T_binary v2` coverage after sampling: `106` tasks, `0` shortfall
- exclusion accounting in current curation pipeline:
  - raw GT pool before video filter: `1,523,422`
  - removed by missing `camera_top` video: `1,041,069`
  - remaining after video filter: `482,353`
  - removed by per-type target/cap sampling: `466,853`
  - final curated: `15,500`
- source snapshot backing the current root release:
  - `previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/benchmark_v1_curated.jsonl`

## 8. Ground-Truth Construction

### 8.1 Overview
All GT labels are automatically constructed from robot-side signals and task metadata; no full manual per-item labeling is used in benchmark generation.

### 8.2 Signal sources by task
- `T1`: `observation.state.effector.effort` + `observation.state.arm.velocity` via trajectory segmentation
- `T2`: `observation.state.effector.effort` contact/release event detection
- `T3`: `action.arm.position` per-step delta (`t -> t+1`) on primary arm (or dominant arm fallback)
- `T4`: `observation.state.arm.velocity` left/right norm activity
- `T6`: `observation.state.arm.velocity` sequence mean over 5-frame context
- `T_temporal`: frame indices plus stage-aware sampling constraints
- `T_binary`: frame index comparison for same-episode frame pairs (segmentation used only for difficulty-stratified sampling)
- `T_progress`: segmented stage label mapped to 5 progress bins

### 8.3 Thresholds and heuristics (current defaults in code)
- shared contact-event detection (`segmentation.py`):
  - effort baseline: median over last 10% frames
  - noise sigma: std over first 10% frames
  - contact when effort `< baseline - 3*sigma` for at least 5 frames
  - release when effort returns to `baseline ± 2*sigma`
- shared sampling visibility guard:
  - for contact episodes: start from `first_contact - 30`
  - no-contact episodes: start from frame row `20`
- `T1`:
  - labels: `pre-approach / approach / contact / hold and carry / transfer / release`
  - `pre-approach` window: `[first_contact-90, first_contact-30)`
  - per-episode per-phase sample budget default: `5`
- `T2`:
  - boundary margin exclusion: `3` frames around contact/release
  - per-episode per-side balanced sampling `contact:no_contact = 1:1`
  - per-class cap default: `5`
- `T3`:
  - static filter: `delta_norm >= 0.005`
  - dominant component min: `>= 0.01`
  - purity filter: `dominant >= 2.0 * second_component`
  - visual-direction mapping for `+x/-x/+y/-y/+z/-z`
  - answer-letter balancing across `A/B/C/D` enabled by default
- `T4`:
  - left/right velocity norms with moving average window `30`
  - active threshold `> 0.3`
  - label set: `both_active / left_only / right_only / both_idle`
- `T6` (current benchmark-v1 path):
  - 5-frame context: offsets `[-6,-3,0,+3,+6]` (`frame_stride=3`, `half_span=2`)
  - sequence-mean speed label:
    - `stationary` if `<= 0.2`
    - `actively_moving` if `>= 0.8`
    - ambiguous middle zone discarded
  - per-episode per-label cap default: `3`
- `T_temporal`:
  - display labels are `X/Y/Z` (not `A/B/C`)
  - requires 3 valid stages and frame spacing `min_gap_frames=30`
  - answer is permutation of display labels sorted by true time
- `T_binary`:
  - two frames from the same episode are rendered into one `composite_panel_v2` comparison image
  - displayed labels are `X/Y`, but the label-to-panel assignment is randomized per item and stored in metadata
  - answer asks which labeled panel happened earlier in time
  - GT is determined only by `frame_index` comparison
  - difficulty strata: `hard_adjacent_stage` and `easy_cross_stage`
  - frame-gap guard: `min_gap_frames=60`, `max_gap_frames=240` in the current root release
- `T_progress`:
  - stage-to-bin map:
    - `0 -> 0-20%`
    - `1 -> 20-40%`
    - `2,3 -> 40-60%`
    - `4 -> 60-80%`
    - `5 -> 80-100%`

### 8.4 Reproducibility scripts
- segmentation core: `benchmark/gt_build/segmentation.py`
- per-task GT: `build_t1_gt.py`, `build_t2_gt.py`, `build_t3_gt.py`, `build_t4_bimanual_gt.py`, `build_t6_gt.py`, `build_t_temporal_gt.py`, `build_t_binary_gt.py`, `build_t_progress_gt.py`
- curation/sampling: `benchmark/gt_build/build_sampling_pipeline.py`
- frame extraction: `benchmark/gt_build/extract_frames.py`
- eval: `benchmark/eval_v1/run_benchmark_v1_eval.py` (wrapper over `run_pilot_eval.py`)
- scoring: `benchmark/eval_v1/score_benchmark_v1.py` (wrapper over `score_pilot.py`)

## 9. Validation

### 9.1 Automatic validation (current completed)
- curation healthcheck (`benchmark_v1_curated_healthcheck.json`):
  - shortfall by type: all `0`
  - missing episode id by type: all `0`
  - per-type answer/label distributions exported
  - per-type zero-task lists exported
- frame extraction (`extract_summary.json`):
  - requested `34,312`, success `34,312`, failed `0`
- curated exact-line duplicate check:
  - duplicate lines in `benchmark_v1_curated.jsonl`: `0`
- eval invalid-handling implemented:
  - parser outputs `INVALID` when cannot parse required format
  - runtime returns `ERROR` and `MISSING_FRAME` where applicable
  - scorer ignores `INVALID/ERROR/MISSING_FRAME` by default (`--keep-invalid` to include)

### 9.2 Manual audit (current status)
- completed targeted audits:
  - `T3` directional sanity checks on manually exported samples (`manual_checks_20260318`), including z-direction wrist-view sanity sample
  - `T_temporal` error-pack generation (`manual_checks_20260325/t_temporal_wrong_review_sample_100.csv`)
- current submission blocker:
  - the multi-annotator audit table is not finalized yet; annotator count, agreement rate, and kappa statistics must be added before paper submission
  - the temporal 100-sample review sheet exists, but the aggregated human-audit summary is not yet finalized in the repository
  - `T_binary` has passed automatic health checks, but no dedicated human audit slice has been added yet
- planned completion criteria before submission:
  - report annotator count and sampling protocol
  - report agreement / Cohen's kappa (or Fleiss' kappa if more than two annotators are used)
  - report per-task audit slices for at least the most ambiguity-prone task families (`T2`, `T3`, `T_temporal`, `T_binary`, and `T_progress`)

### 9.3 Known task-specific caveats (current v1)
- `T2`: visual observability may be weaker near contact boundaries; mitigation uses margin filtering but ambiguity remains.
- `T3`: in current full-model diagnostics, accuracy is close to/below random for some VLMs; this is treated as a hard diagnostic task.
- `T4`: current source JSONL stores `label_id` without native `question/choices/answer`; eval script injects canonical question/options during normalization.
- `T6`: measurable signal exists, but model predictions show stationary-prior tendency in diagnostics.
- `T_temporal`: protocol-sensitive; strong output-order bias observed in Qwen full-run diagnostics, so not recommended as sole headline metric in v1.
- `T_binary`: included in the current root release via `composite_panel_v2`; automatic balance/coverage checks pass, but Qwen smoke diagnostics still show residual left-panel bias and full multi-model validation is not finalized yet.

## 10. Evaluation Protocol

### 10.1 Standard protocol
- input:
  - item JSONL: `benchmark_v1_curated.jsonl`
  - frame dir: `benchmark_v1_frames_tbinary_20260330`
- prompting:
  - multiple-choice tasks: answer with single letter
  - `T_temporal`: answer with 3-letter permutation over item-provided labels (`X/Y/Z` by default)
- parsing:
  - robust parser supports direct labels, cue-based text, and structured temporal outputs
  - unresolved parse returns `INVALID`
- runtime failure handling:
  - missing local frames -> `MISSING_FRAME`
  - API/request failures after retries -> `ERROR`
- scoring:
  - default excludes `INVALID/ERROR/MISSING_FRAME` from denominator
  - random baselines in benchmark scorer:
    - `T1=0.25`
    - `T2=0.50`
    - `T3=0.25`
    - `T4=0.25`
    - `T6=0.50`
    - `T_temporal=0.167`
    - `T_binary=0.50`
    - `T_progress=0.20`

### 10.2 Required reporting (recommended minimum)
- per-task accuracy with valid-N
- random-baseline delta by task
- overall valid-only accuracy + invalid count
- arm-type breakdown (and explicit unknown bucket if schema has missing arm_type)
- per-task distribution sanity (answer/label histograms)

### 10.3 Recommended auxiliary analyses
- class-balanced metrics for `T2/T6`
- protocol sensitivity and order-bias analysis for `T_temporal`
- item-level error taxonomy for `T3` and `T_temporal`
- common-vs-long-tail breakdown by task families (where metadata is available)

## 11. Limitations and Threats to Validity

### 11.1 Data source limitations
- **Single source dataset**: all items are derived from GM-100 only.  
  *Mitigation: benchmark reporting is explicitly framed as within-distribution diagnosis rather than cross-dataset generalization.*
- **Single robot platform subset**: v1 uses the Agilex Cobot Magic subset only.  
  *Mitigation: task-family diversity and arm-type coverage partially compensate for platform homogeneity; cross-platform extension is reserved for v2.*
- **Single-lab collection style**: demonstrations come from one collection environment and teleoperation setup.  
  *Mitigation: the benchmark avoids broad transfer claims and requires per-task reporting instead of a single headline number only.*
- **Success-demonstration skew**: the source trajectories are successful demonstrations, not failure-rich datasets.  
  *Mitigation: tasks such as `T2` still introduce intra-trajectory contrast (contact vs no-contact), but failure understanding remains out of scope for v1.*
- **Top-view-first benchmark design**: v1 is defined around `camera_top` rather than a multi-view protocol.  
  *Mitigation: this reduces view variance in v1; multi-view expansion is treated as a future benchmark revision rather than an implicit current claim.*

### 11.2 Label construction limitations
- **Threshold sensitivity in contact/velocity segmentation**: several tasks depend on hand-tuned thresholds over effort or velocity.  
  *Mitigation: thresholds are centralized, documented, and applied consistently across the dataset; targeted manual checks are used on known hard cases.*
- **Ambiguity near state-transition boundaries**: labels close to contact, release, or transfer boundaries can be visually ambiguous.  
  *Mitigation: the pipeline uses boundary guards, stage-aware sampling windows, and exclusion of known ambiguous regions where applicable.*
- **Robot-signal-derived labels do not always perfectly match visual observability**: a physically defined event is not always equally visible from the top camera.  
  *Mitigation: task caveats are documented explicitly, and weak-signal tasks are not interpreted as standalone evidence of general manipulation competence.*

### 11.3 Evaluation limitations
- **Zero-shot-first prompt protocol**: current evaluation emphasizes a fixed prompt rather than multi-prompt optimization.  
  *Mitigation: this keeps the protocol stable across models, but prompt sensitivity analysis remains recommended auxiliary work.*
- **English prompt focus**: current scripts are written for English prompts only.  
  *Mitigation: the benchmark is framed as an English evaluation protocol in v1; multilingual prompting is future work.*
- **API/backend instability can produce small invalid/error residue**: model serving errors and parse failures can occur.  
  *Mitigation: the runner retries requests, records `ERROR` / `INVALID` / `MISSING_FRAME`, and the scorer reports valid-only accuracy by default.*

### 11.4 Interpretation limitations
- **Overall score is not sufficient alone**: per-task differences can be larger than overall differences.  
  *Mitigation: the recommended reporting protocol requires task-level and arm-type breakdowns.*
- **Strong score on one family does not imply broad manipulation competence**: models may specialize in motion-state cues while failing on direction, contact, or temporal ordering.  
  *Mitigation: score interpretation is intentionally diagnostic and multi-axis rather than leaderboard-only.*
- **Benchmark score is not a deployment guarantee**: visual benchmark success does not imply closed-loop VLA success or safety.  
  *Mitigation: out-of-scope uses are stated explicitly and causal VLA-transfer claims are disallowed without separate control experiments.*

## 12. Responsible AI and Data Governance

### 12.1 Data provenance and personal data
- source: GM-100 Cobot Magic trajectories
- v1 benchmark artifacts are robot-scene frames and metadata fields for evaluation
- no explicit personal-identifying fields are used in benchmark JSONL
- governance/consent obligations inherit from the upstream GM-100 data release terms

### 12.2 Potential risks
- over-claiming VLM-to-VLA transfer
- overfitting to one view/lab/platform style
- misuse as generic VLM ranking leaderboard

### 12.3 Potential benefits
- clearer diagnosis of robotic visual perception failures
- stronger benchmark transparency (task caveats + protocol constraints)
- better comparability across VLMs for manipulation-relevant visual cues

### 12.4 Maintenance plan (current)
- v1.x bugfix scope:
  - parser robustness fixes
  - path/config consistency fixes
  - metadata/schema consistency fixes
- v2 redesign scope:
  - substantial task-definition changes
  - new modalities/views
  - major sampling strategy changes
- issue handling channel:
  - repository issues (URL TBD)

### 12.5 Machine-readable metadata and submission compliance
- a Croissant metadata record must accompany benchmark release and paper submission
- the release package should include both core dataset fields and Responsible AI fields, rather than relying only on the markdown card
- Hugging Face can expose baseline Croissant metadata, but benchmark-specific Responsible AI fields and any custom provenance details should be checked and completed manually before submission
- current status: the markdown card contains the intended policy content, but the submission-ready Croissant artifact path is still pending in the repository

## 13. Release Notes

### v1.1 (current root working release, 2026-03-31)
- promoted the root `benchmark_v1_curated.jsonl` to the latest 8-family release (`15,500` items)
- adopted `T_binary v2` in the root release with `composite_panel_v2` presentation and `X/Y` answer protocol
- updated the root frame cache reference to `benchmark_v1_frames_tbinary_20260330` (`34,312` images, `0` failures)
- updated default eval/sampling/pipeline entrypoints to the latest source snapshot under `previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/`
- archived the former 7-family root release under `previous_results/root_release_snapshots/benchmark_v1_root_7family_20260331_before_latest_promotion/`

### v1.0 (historical 7-family root freeze, 2026-03-26)
- 7-family benchmark curation pipeline finalized (`14,000` items)
- top-view frame extraction completed (`31,382` images, `0` failures)
- benchmark eval/scoring wrappers consolidated under `eval_v1/`
- `T_temporal` protocol hardened to `X/Y/Z` labels and robust parser
- documented caveats for `T3`, `T6`, and `T_temporal`

## 14. Preliminary Diagnostic Results

For full evaluation results across multiple models, see the accompanying paper and release artifacts.

A complete full-model rerun on the **current root release** (`15,500` items, including `T_binary v2`) is still pending. The previously reported single-model Qwen sanity-check run was produced on the historical 7-family root release and should not be interpreted as a result for the current root release.

Legacy 7-family sanity-check result path:
- `benchmark/eval_results_v1/benchmark_v1_qwen3vl8b_instruct_full.jsonl`

Current-status note:
- use the latest root release for all new evaluations
- rerun preliminary diagnostics after root-promotion before citing any headline accuracy in the card or paper

## 15. Citation

```bibtex
@misc{bimanip_bench_v1_2026,
  title        = {BiManip-Bench: A Manipulation-Relevant Vision Benchmark for Bimanual Robot Trajectories},
  author       = {TBD},
  year         = {2026},
  howpublished = {\url{TBD}},
  note         = {Version 1.1-draft}
}
```

## 16. Contact

Maintainer: TBD  
Repository issues: TBD
