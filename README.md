# Robotic Manipulation Process Understanding Benchmark

A comprehensive benchmark for **process-aware robotic manipulation understanding**. This benchmark evaluates whether models can understand manipulation episodes beyond object recognition—focusing on process state, temporal reasoning, and action chains.

## Quick Overview

**Goal**: Evaluate robotic manipulation understanding through 12 question categories covering state recognition, motion analysis, and outcome prediction.

**Scope**: 57,892 items across 4 datasets (GM-100, RH20T, REASSEMBLE, AIST-Bimanual)

**Split**: 85% SFT / 15% Eval with strict episode-level isolation

**Key Questions**:
- What process phase is the manipulation in?
- Has effective contact been established?
- What is the dominant motion direction?
- Will this attempt succeed?
- What low-level action comes next?

## Benchmark Design

### Core Principless

**Process-aware, not object-only**: Evaluates manipulation in terms of state, motion, temporal order, progress, and action chains—not just object categories.

**Dataset-native ground truth**: Labels constructed from native signals (force/torque, gripper state, TCP pose, velocity patterns) rather than free-form annotation, keeping the benchmark scalable and reproducible.

**Episode-level isolation**: Strict train/eval split at the episode level—same recording never appears in both splits.

## Question Categories

12 question categories organized into two families:

### Static Process Monitoring

Tasks that identify current state from single frames or short clips:

| ID | Task | Question | Input |
| --- | --- | --- | --- |
| `T1` | Phase Recognition | What coarse process phase is the manipulation in? | Single frame |
| `T2` | Contact Detection | Has effective contact been established? | Single frame |
| `T4` | Bimanual Coordination State | How are the two arms currently coordinating? | Short clip |
| `T10` | Current Primitive Recognition | What low-level action is being executed now? | Short clip |

### Dynamic Process Reasoning

Tasks requiring temporal reasoning, progress estimation, or outcome prediction:

| ID | Task | Question | Input |
| --- | --- | --- | --- |
| `T3` | Motion Direction Prediction | What is the dominant motion direction? | Short clip |
| `T5` | Primitive-local Progress | Is the current step early, middle, or late? | Short clip |
| `T6` | Motion State Recognition | Is the manipulator actively moving or stationary? | Short clip |
| `T7` | Operation Outcome Prediction | Will this attempt eventually succeed? | Early clip |
| `T8` | Temporal Ordering | What is the correct order of three shuffled frames? | Three frames |
| `T9` | Temporal Priority Prediction | Which of two frames happened earlier? | Two frames |
| `T11` | Next Primitive Prediction | What low-level action should occur next? | Short clip |
| `T12` | Primitive Chain Restoration | Which action fills the masked slot in the chain? | Local chain |

**Note**: Legacy code names: `T_progress` = `T5`, `T_temporal` = `T8`, `T_binary` = `T9`

## Primitive Semantics

Shared primitive semantics are anchored to the **REASSEMBLE** low-level action vocabulary—the only source providing native fine-grained action annotations.

### Canonical Low-level Primitives

9 core primitives used in `T10`, `T11`, `T12`:

| Primitive | Meaning |
| --- | --- |
| `Approach` | Move toward target before stable contact |
| `Align` | Orient or position relative to target |
| `Grasp` | Establish stable hold on object |
| `Lift` | Raise object after grasp |
| `Pull` | Draw object outward or backward |
| `Push` | Move object by direct contact |
| `Release` | Terminate hold or contact |
| `Twist` | Apply rotational manipulation |
| `Nudge` | Make small corrective adjustment |

**Important**: We do not force dense primitive labels onto datasets that don't natively support them. Outside REASSEMBLE, coarser process states are derived from dataset-native signals for `T1-T9`.

### REASSEMBLE High-level Actions

Above primitives, REASSEMBLE provides:
- **4 verbs**: `pick`, `place`, `insert`, `remove`
- **17 object categories**: BNC, D-SUB, Ethernet, USB, bolts, gears, pegs, etc.
- **68 effective task actions**

This creates a clean hierarchy:
- **Low-level**: How the manipulation is executed (primitive)
- **High-level verb**: What kind of task step (pick/place/insert/remove)
- **Object**: Which assembly part

## Dataset Composition

### Data Sources

Four datasets contribute to the benchmark. Each provides only the task types it natively supports:

| Dataset | Role | Supported Categories | Items |
| --- | --- | --- | ---: |
| **GM-100** | Goal-conditioned; single/bimanual process | T1, T2, T3, T4, T5, T6, T8, T9 | 15,500 |
| **RH20T** | Force/torque + gripper + TCP grounded | T1, T2, T3, T5, T6, T7, T8, T9 | 15,800 |
| **REASSEMBLE** | Action-chain + primitive labels | T1, T2, T5, T6, T7, T8, T9, T10, T11, T12 | 17,165 |
| **AIST-Bimanual** | Bimanual kinematics (no force) | T3, T4, T6, T8, T9 | 9,427 |
| **Total** | | | **57,892** |

### Task Coverage

| Dataset | Distinct Tasks | Split Groups |
| --- | ---: | ---: |
| GM-100 | 101 | 3,558 |
| RH20T | 76 | 507 |
| REASSEMBLE | 31 | 31 |
| AIST-Bimanual | 10 | 170 |

*Group semantics: episode_id for GM-100, recording_id for others*

## Ground Truth Construction

### General Principles

1. Prefer dataset-native supervision over free-form annotation
2. Define task-specific local decision units before sampling
3. Use deterministic builders for labels
4. Manual audit only for calibrating ambiguous mappings
5. Keep object grounding and primitive pairs as auxiliary layers

### Category-Specific Details

#### T1 - Phase Recognition
**Goal**: Classify current coarse process phase
**Sources**: Task-stage intervals, process segmentation
**Sampling**: Single frame from stable phase interval
**Labels**: pre-contact, manipulation, release, or dataset-specific phases

#### T2 - Contact Detection
**Goal**: Determine if effective contact established
**Sources**: Force/torque onset, gripper closure, state metadata
**Sampling**: Single frame from contact-positive or negative regions

#### T3 - Motion Direction Prediction
**Goal**: Infer dominant motion direction from clip
**Sources**: Planar displacement, velocity components, calibrated mappings
**Important**: Direction labels calibrated per dataset/viewpoint, not raw coordinates
**Offsets**: GM-100 `[-10,-5,0,5]`, RH20T `[-20,-10,0,10]`, AIST-Bimanual `[-30,-15,0,15]`

#### T4 - Bimanual Coordination State
**Goal**: Determine which arms are active
**Sources**: Per-arm motion energy, TCP/velocity thresholds
**Labels**: both-active, left-only, right-only, idle

#### T5 - Primitive-local Progress
**Goal**: Determine if manipulation step is early, middle, or late
**Formula**: `u = (t - start) / (end - start)`
**Avoids**: Ambiguous five-way percentages
**Unit Definition**:
  - GM-100: contact-aware local interval
  - RH20T: force/torque + gripper + TCP synthesized interval
  - REASSEMBLE: official successful high-level segment

#### T6 - Motion State Recognition
**Goal**: Is manipulator actively moving or stationary?
**Sources**: Velocity magnitude, motion-energy thresholding

#### T7 - Operation Outcome Prediction
**Goal**: Will the attempt eventually succeed?
**Sources**: Native success/failure metadata, completion ratings
**Supported**: RH20T, REASSEMBLE only
**Sampling**: Early segments to predict future outcome

#### T8 - Temporal Ordering
**Goal**: Order three shuffled frames correctly
**Sources**: Frame timestamps or indices
**Sampling**: Three frames from same episode, then permuted

#### T9 - Temporal Priority Prediction
**Goal**: Which frame happened earlier?
**Sources**: Frame index or timestamp order
**Presentation**: Often rendered as two-panel comparison

#### T10 - Current Primitive Recognition
**Goal**: Identify current low-level action
**Source**: REASSEMBLE only
**Answer Space**: `Approach, Align, Grasp, Lift, Pull, Push, Release, Twist, Nudge`
**Sampling**: Short clip within labeled action segment

#### T11 - Next Primitive Prediction
**Goal**: Predict next action given context
**Source**: REASSEMBLE only
**Answer Space**: Same canonical primitive set

#### T12 - Primitive Chain Restoration
**Goal**: Fill masked step in action chain
**Source**: REASSEMBLE only
**Answer Space**: Same canonical primitive set
**Sampling**: Local chain templates with one masked slot

## Temporal Context Protocol

No universal frame rule—each category uses the minimal context required:

| Category | Context Type |
| --- | --- |
| T1, T2 | Single frame |
| T3, T4, T5, T6, T7, T10, T11, T12 | Short ordered clip |
| T8 | Three shuffled frames |
| T9 | Two-frame comparison |

Clip spacing chosen to match task requirements:
- **T3**: Wider motion-sensitive offsets
- **T5**: Local-step-centered short clips
- **T7**: Wider spacing to encourage future-outcome prediction

## Train/Eval Split

**Version**: `splits_v1` (frozen)

**Ratio**: 85% SFT / 15% Eval

**Isolation**: Episode-level—same recording never appears in both splits

### Split Breakdown

| Dataset | Eval | SFT | Total |
| --- | ---: | ---: | ---: |
| GM-100 | 2,643 | 12,857 | 15,500 |
| RH20T | 2,422 | 13,378 | 15,800 |
| REASSEMBLE | 2,562 | 14,603 | 17,165 |
| AIST-Bimanual | 1,424 | 8,003 | 9,427 |
| **Total** | **9,051** | **48,841** | **57,892** |

### Split Strategy

The split is not purely random—fragile categories (`T10/T11/T12`) are upweighted in eval selection to prevent them from being washed out by larger-volume tasks.

## Evaluation & Quality Assurance

### Evaluation Pipeline

**Dataset-wise evaluation** (not monolithic)—each dataset uses its own frame cache and naming conventions

**Key scripts**:
- `benchmark/eval_v1/run_all_eval_sets.py`
- `benchmark/eval_v1/score_all_eval_sets.py`

**Flow**:
1. Read `benchmark/splits_v1/eval_manifest.json`
2. Run each dataset split separately
3. Write per-dataset results
4. Aggregate into one summary

### Manual Audit & Calibration

Manual effort is **focused on high-leverage calibration**, not exhaustive relabeling:

- Benchmark validity checks
- Direction mapping calibration for `T3`
- Primitive-chain calibration
- Fragile-category inspection (`T5`, `T7`, `T10`, `T11`, `T12`)

Three audit variants available:
- **`human_audit`**: Full audit (2.5% sampling)
- **`human_audit_debug`**: Debugging variant (2.5% sampling)
- **`human_audit_smoke`**: Quick smoke test (0.1% sampling, 4 samples per task)

## Scope & Limitations

### What's Included

Fully supported by: GM-100, RH20T, REASSEMBLE, AIST-Bimanual

All 12 question categories, with dataset-specific coverage as shown in tables above.

### Known Limitations

1. **REASSEMBLE**: Only 37 recordings → `T10/T11/T12` are statistically more fragile
2. **AIST-Bimanual**: No force signal → motion/coordination tasks only, not contact-grounded reasoning
3. **RH20T**: Fixed primary camera for current release → viewpoint bias should be noted
4. **RoboMIND2.0**: Not yet integrated into paper-facing loop (future work)

## Key Resources

Most relevant files for understanding the benchmark:

- `benchmark/NEURIPS_ED_MASTER_PLAN.md`
- `benchmark/ARCHITECTURE_MAP.md`
- `benchmark/splits_v1/split_summary.json`
- `benchmark/splits_v1/eval_manifest.json`
- `benchmark/splits_v1/all_sft_merged.jsonl`
- `benchmark/splits_v1/all_eval_merged.jsonl`
