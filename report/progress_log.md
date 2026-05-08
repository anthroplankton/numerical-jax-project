# Project Progress Log

## Phase 1: Initial Python/JAX Scaffold

- Date / phase label：初始 scaffold；對應 git commit `4f6001b chore: bootstrap initial JAX project scaffold`
- What was done：
  - 建立 course project repository 的基本 Python project structure。
  - 使用 `uv`、`pyproject.toml`、`uv.lock` 管理 project 與 dependencies。
  - 建立 `src/jax_tpu_project/` package。
  - 建立 JAX runtime/device summary utility and CLI。
  - 加入 `scripts/check_jax_device.sh` 作為本機 JAX device sanity check。
  - 加入 pytest runtime tests。
  - 建立 `.gitignore`，排除 caches、virtual environments、credentials、generated runs、raw data、model weights、uncurated cloud outputs。
- Files or modules added：
  - `AGENTS.md`
  - `README.md`
  - `pyproject.toml`
  - `uv.lock`
  - `.gitignore`
  - `scripts/check_jax_device.sh`
  - `src/jax_tpu_project/__init__.py`
  - `src/jax_tpu_project/cli.py`
  - `src/jax_tpu_project/runtime.py`
  - `tests/test_runtime.py`
- Commands/checks known from repository：
  - README documents `bash scripts/check_jax_device.sh`
  - README documents `uv run python -m jax_tpu_project.cli devices`
  - `pyproject.toml` configures pytest test path as `tests`
- Current evidence/results：
  - Runtime utility returns JSON-serializable backend/device information.
  - Tests check `default_backend` and visible device fields.
- Limitations：
  - Scaffold only; no training demo in this phase.
  - No cloud TPU workflow or benchmark evidence.
- Next planned step：
  - Add a small raw-JAX training-oriented demo suitable for local smoke testing and future TPU comparison.

## Phase 2: Demo 1 Raw-JAX CNN Benchmark Foundation

- Date / phase label：CNN benchmark foundation；對應 git commit `b4c8abf feat: add hand-written MNIST CNN benchmark`
- What was done：
  - 建立 hand-written CNN benchmark for MNIST-shaped image classification。
  - 使用 raw JAX primitives，不使用 Flax、Optax、PyTorch、TensorFlow。
  - 實作 deterministic synthetic MNIST-shaped data generator。
  - 實作 CNN parameter initialization、forward pass、cross-entropy loss、accuracy、`jax.jit` training step、explicit SGD update。
  - 實作 benchmark runner，包含 warmup steps、timed steps、terminal logging、JSON metrics output。
  - 加入 CLI example script `examples/cnn_mnist_benchmark.py`。
  - 更新 README with Demo 1 commands and expected metrics。
  - 加入 CNN tests for shapes, forward pass, training smoke behavior, metrics writing, and deterministic seed behavior。
- Files or modules added/updated：
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
  - `tests/test_cnn_mnist.py`
  - `src/jax_tpu_project/__init__.py`
  - `README.md`
- Commands/checks known from repository：
  - Quick smoke benchmark:
    ```bash
    uv run python examples/cnn_mnist_benchmark.py \
      --dataset synthetic \
      --steps 3 \
      --batch-size 16 \
      --seed 0 \
      --output-dir runs/smoke \
      --platform-label local
    ```
  - Longer local benchmark:
    ```bash
    uv run python examples/cnn_mnist_benchmark.py \
      --dataset synthetic \
      --steps 50 \
      --batch-size 64 \
      --learning-rate 0.05 \
      --warmup-steps 5 \
      --seed 0 \
      --output-dir runs/local-cnn-mnist \
      --platform-label local
    ```
- Current evidence/results：
  - Existing ignored generated artifact: `runs/smoke/cnn_mnist_metrics.json`。
  - That artifact records `dataset_name: synthetic`, `backend: cpu`, `platform_label: local`, `steps: 3`, `warmup_steps: 1`, `batch_size: 16`, `seed: 0`。
  - The artifact includes loss, accuracy, total training time, average step time, examples per second, backend, devices, and output artifact path。
- Limitations：
  - Current completed benchmark data path is synthetic only。
  - `mnist` and `fashion_mnist` are accepted as CLI choices but currently reserved for future local-file dataset support and raise `FileNotFoundError`。
  - No Google Cloud TPU run has been completed in this repository。
  - No cloud monitoring screenshots, TPU logs, or local-vs-TPU comparison artifacts exist yet。
  - Existing `runs/` artifact is ignored and should not be treated as curated final report evidence until copied or summarized intentionally under `report/results/` with context。
- Next planned step：
  - Implement real MNIST/Fashion-MNIST local-file loading, then produce curated local benchmark artifacts before moving to TPU execution.

## Phase 3: Presentation Planning and Current Status Documentation

- Date / phase label：2026-05-08 presentation planning documentation
- What was done：
  - Created `report/` for course-report materials.
  - Added and revised a Traditional Chinese presentation plan with a broader three-demo roadmap.
  - Added this progress log grounded in repository files, git history, tests, and available generated metrics.
  - Added a current project status summary for local demo usage, tested behavior, missing work, and next milestones.
  - Clarified that Demo 2 and Demo 3 are planned work, not implemented work.
- Files or modules added：
  - `report/presentation_plan.md`
  - `report/progress_log.md`
  - `report/current_status.md`
- Commands/checks run during this documentation task：
  - `rg --files`
  - `git status --short`
  - `git log --oneline --decorate -10`
  - `git show --stat --oneline --decorate HEAD`
  - `git show --stat --oneline --decorate 4f6001b`
  - Inspected README, pyproject, source modules, examples, tests, scripts, `.gitignore`, and `runs/smoke/cnn_mnist_metrics.json`
- Current evidence/results：
  - Documentation now separates completed local scaffold/Demo 1 CNN foundation from planned real dataset work, planned TPU benchmark, planned Demo 2, and optional planned Demo 3。
- Limitations：
  - This phase is documentation-only。
  - No new source code, tests, cloud workflows, dependencies, Docker files, CI, or notebooks were added。
  - No real MNIST/Fashion-MNIST, TPU, Demo 2, or Demo 3 results are claimed。
- Next planned step：
  - Run or curate a local Demo 1 benchmark result under `report/results/`, then implement real dataset loading and prepare TPU workflow documentation.

## Planned Phases

### Phase 4: Real MNIST/Fashion-MNIST and Curated Local Result

- Status：planned, not completed。
- Goal：extend Demo 1 from synthetic MNIST-shaped data to real MNIST and Fashion-MNIST local-file data。
- Planned work：
  - Add local-file dataset loading without making network access required for default tests。
  - Keep synthetic mode as the default smoke path。
  - Produce a curated local result under `report/results/` with command, environment, configuration, and metrics。
  - Add or update tests for small dataset-loading behavior where practical。
- Evidence needed before marking complete：
  - Source code for loader。
  - Tests or smoke checks。
  - Curated local benchmark artifact and documented command。

### Phase 5: TPU Workflow and Local-vs-TPU Comparison for Demo 1

- Status：planned, not completed。
- Goal：run Demo 1 on Google Cloud TPU and compare against local execution using reproducible evidence。
- Planned work：
  - Document TPU VM setup, repository checkout/update, environment setup, run command, monitoring, output retrieval, and cleanup。
  - Run CNN benchmark with a clear `platform_label` such as `tpu`。
  - Capture JSON metrics, terminal logs, backend/devices, and monitoring evidence where available。
  - Generate local-vs-TPU comparison table or figure。
- Evidence needed before marking complete：
  - TPU run metrics。
  - Cloud logs or screenshots。
  - Cleanup notes。
  - Comparison artifact grounded in saved metrics。

### Phase 6: Demo 2 Existing JAX Training Stack

- Status：planned, not completed。
- Goal：compare Demo 1 raw-JAX implementation with a higher-level JAX ecosystem workflow。
- Planned work：
  - Choose Flax + Optax or a small existing JAX model/example。
  - Document what abstractions are introduced and which code becomes simpler。
  - Reuse Demo 1 metrics JSON format where practical。
  - Explore whether the same local vs TPU workflow can be reused。
  - Add lightweight local smoke tests。
- Evidence needed before marking complete：
  - Demo 2 source/example code。
  - Documented run command。
  - Metrics output。
  - Tests or smoke checks。
  - Clear comparison against Demo 1。

### Phase 7: Optional Demo 3 Pretrained/Gemma Cloud Demo

- Status：optional planned work, not completed。
- Goal：show how a pretrained-model cloud workflow differs from the smaller training demos, only if access and scope are manageable。
- Planned work：
  - Review model source, license, access requirements, and usage constraints。
  - Decide whether the demo is inference-only, fine-tuning, or workflow demonstration。
  - Estimate dependencies, hardware, memory, quota, and setup time。
  - Prepare fallback material if access, quota, hardware, or time is unavailable。
- Evidence needed before marking complete：
  - Model selection and access/license notes。
  - Setup and run commands。
  - Cloud or local execution evidence。
  - Clear fallback or limitation notes if not executed。
