# Reproducibility Drift and GPU Configuration Impact

This note documents expected analysis drift across branch generations and isolates
what drift is likely due to GPU configuration/orchestration versus core
modernization changes.

## Scope

Two comparisons are covered:

1. `master` (2018-era baseline) -> `threshold-tweaks` (modernized runtime path)
2. `threshold-tweaks` -> `multi-gpu` (GPU orchestration and scheduling changes)

The goal is to set expectations for what can change, why it can change, and what
is most likely to affect final figures/metrics.

## Comparison 1: `master` (2018) vs `threshold-tweaks`

### Expected drift: **Moderate to High**

`threshold-tweaks` modernizes multiple layers of the stack that can change model
training trajectories and numeric outputs even when the analysis intent is the
same.

### Why drift is expected

- **TensorFlow/Keras compatibility layer**  
  The code path now runs through TF1-compat wrappers under modern TF runtimes.
  Numeric kernels, execution order, and backend behavior may differ from 2018.

- **AnnData/Scanpy modernization and dense conversion logic**  
  Legacy sparse/view handling (`.X.A`, implicit view behavior) was replaced with
  explicit conversion utilities. This can alter memory layout, ordering of
  operations, and edge-case handling.

- **Batch-size policy changes**  
  Several training defaults moved away from original 2018 settings (e.g. larger
  defaults in modernization branches). Batch size materially affects optimizer
  dynamics and learned parameters.

- **I/O and pipeline hygiene updates**  
  Output-directory guarantees, reconstruction skipping, and orchestration updates
  reduce failure modes but can change when and how outputs are regenerated.

- **Environment and dependency pinning updates**  
  New package versions, CUDA stack, and platform assumptions can produce small to
  large numerical differences compared with historical 2018 environments.

### Practical interpretation

- Relative biological signal and qualitative conclusions may still align.
- Exact numerical parity with 2018 artifacts is not expected.
- Treat `threshold-tweaks` as a reproducible *modern baseline*, not a bitwise
  recreation of 2018 outputs.

## Comparison 2: `threshold-tweaks` vs `multi-gpu`

### Branch state note

At the branch-tip level, `threshold-tweaks...multi-gpu` may show no committed
difference if changes are currently in the working tree. The analysis below
covers the effective `multi-gpu` code path currently implemented in this repo:

- `code/ModelTrainer.py`
- `code/scgen/gpu_utils.py`
- `code/vec_arith_pca.py`
- `code/train_scGen.py`

### Expected drift from GPU/orchestration: **Low to Moderate (reduced by seed + determinism mitigations)**

Core model definitions are mostly unchanged; however, execution order and
concurrency changed substantially, which can alter stochastic training outcomes.

### Seed policy note (intentional)

This codebase intentionally does **not** enforce a globally unique per-task seed
across the entire end-to-end run. The reason is historical fidelity: 2018 runs
were not tightly seed-controlled, and imposing strict globally unique seed
structure can introduce an artificial execution discipline that did not exist in
the original workflow.

Current behavior therefore targets a middle ground:
- provide lightweight deterministic controls to reduce avoidable drift,
- while avoiding an over-structured seed scheme that could move behavior farther
  from the historical stochastic profile.

### What changed and impact

- **Parallel stage scheduling in `ModelTrainer all`**
  - Stage 1 now schedules `vec_arith_pca`, `vec_arith`, `st_gan`, `train_cvae`
    with GPU-count-aware allocation.
  - Stage 2 runs `train_scGen` with multi-GPU visibility.
  - **Impact:** Different process start timing and resource contention can change
    RNG usage and floating-point accumulation paths.

- **Freed-device-aware GPU reuse**
  - Scheduler now reassigns work to GPUs as jobs finish, reducing uneven
    contention.
  - **Impact:** Better utilization and fewer pathological waits; also reduces
    run-to-run variance caused by accidental oversubscription hotspots.

- **`vec_arith_pca` train/train parallel then reconstruct**
  - Two training variants now run concurrently before reconstruction.
  - **Impact:** Same logical dependency, but concurrent execution can shift
    low-level numeric behavior and timing.

- **`train_scGen` phased parallelization**
  - Phase 1: parallel train
  - Phase 2: parallel reconstruct
  - Phase 3: parallel heldout-train
  - **Impact:** Increased throughput; potential stochastic drift due to process
    concurrency and GPU scheduling.

- **Run manifest capture**
  - `ModelTrainer all` now records commit, environment snapshot,
    `CUDA_VISIBLE_DEVICES`, seed-related env vars, and command plan.
  - **Impact:** Improves auditability and post-hoc attribution of drift; does not
    by itself enforce determinism.

- **Lightweight stochastic drift mitigations (implemented)**
  - Launcher now injects stable per-process seeds derived from `SCGEN_SEED` into
    `SCGEN_PROCESS_SEED`, `PYTHONHASHSEED`, `NUMPY_SEED`, and `TF_SEED`.
  - Determinism flags are enabled by default (can be disabled with
    `SCGEN_ENABLE_DETERMINISM=0`): `TF_DETERMINISTIC_OPS=1`,
    `TF_CUDNN_DETERMINISTIC=1`.
  - Stochastic entry scripts now call a shared seed hook so Python `random`,
    NumPy, and TensorFlow are seeded consistently per process.
  - **Impact:** Materially lowers stochastic drift for repeated runs under the
    same environment and command plan.

### Closer or farther from the original 2018 analysis?

- **`master (2018)` -> `threshold-tweaks`: farther overall**
  - Modern runtime, dependency, and data-handling changes remain the dominant
    source of drift from original 2018 outputs.
- **`threshold-tweaks` -> current `multi-gpu`: mixed, but net closer in stability**
  - Parallelization introduces potential drift pressure.
  - The new seed/determinism mitigations plus freed-device-aware scheduling push
    results toward *more stable and repeatable* behavior across reruns.
  - Net effect: still not 2018-identical, but generally closer to a consistent
    baseline than unseeded multi-GPU orchestration.

## What is likely to shift analysis most?

1. **Modernization layer (`master` -> `threshold-tweaks`)**: largest expected
   source of drift.
2. **Parallel/stochastic execution (`threshold-tweaks` -> `multi-gpu`)**:
   secondary source; now reduced by deterministic seeding and runtime flags.
3. **GPU assignment policy itself** (freed-device-aware vs naive round-robin):
   mostly affects utilization/stability; smaller direct effect on scientific
   outcomes than stochastic parallelism.

## Recommended interpretation framework

When comparing outputs across these generations:

- Expect **qualitative consistency** where biology is strong.
- Allow **quantitative tolerance bands** for metric/embedding differences.
- Attribute drift in this order:
  1. environment/runtime modernization,
  2. batch-size and training-policy differences,
  3. parallel orchestration and GPU scheduling.

## Controls to strengthen reproducibility claims

- Keep deterministic seed env vars stable between runs (`SCGEN_SEED`,
  `SCGEN_PROCESS_SEED`, `PYTHONHASHSEED`, `NUMPY_SEED`, `TF_SEED`).
- Keep determinism flags enabled (`TF_DETERMINISTIC_OPS=1`,
  `TF_CUDNN_DETERMINISTIC=1`) unless profiling requires otherwise.
- Keep and archive generated run manifests under `data/run_manifests/` (includes
  command plan, env, seeds, and commit).
- Re-run key analyses multiple times and report spread (not only one run).
