# 手寫 CNN on MNIST / Fashion-MNIST：Local vs Google Cloud TPU Benchmark 簡報計畫

## 專案摘要

本專案是「Numerical Computation with JAX」課程專題，整體方向是建立一組可展示、可測量、可延伸到 Google Cloud TPU 的 JAX workflow。第一個 demo 以 raw JAX 手寫 CNN training benchmark 為核心，目前已完成 local scaffold、JAX runtime/device sanity check、synthetic MNIST-shaped data smoke benchmark、JSON metrics output 與相關 pytest smoke tests。後續規劃會延伸到三個 demo：Demo 1 完成 MNIST/Fashion-MNIST 的 local vs TPU benchmark；Demo 2 用 Flax + Optax 或既有 JAX training stack 對照 raw-JAX 寫法；Demo 3 在條件允許時展示 Gemma-like 或其他 pretrained-model cloud workflow。現階段尚未完成 real MNIST/Fashion-MNIST loader、TPU execution、cloud monitoring、local-vs-TPU comparison、Demo 2 implementation 或 Demo 3 implementation。

## 簡報目標

- 內容安排會先用簡短背景說明 JAX backend、TPU execution、benchmark metrics 等脈絡，再進入目前實作、可展示的 local benchmark，以及後續雲端與模型 demo 規劃。
- 簡報目標是誠實呈現目前已完成的 local raw-JAX CNN benchmark foundation，說明它如何產生可重現的 metrics 與測試證據，並把 Demo 1、Demo 2、Demo 3 的完成狀態與下一步清楚分開。

## Three-Demo Roadmap

### Demo 1: Hand-Written Model

- Topic：CNN on MNIST / Fashion-MNIST。
- Goal：用 raw JAX 實作 CNN training workflow，完成 local vs Google Cloud TPU training benchmark。
- Current status：local raw-JAX CNN benchmark foundation exists。
- Current data path：目前完成的是 deterministic synthetic MNIST-shaped data；除非 repository 後續新增 loader，否則 real MNIST/Fashion-MNIST 尚未完成。
- Evidence now：`src/jax_tpu_project/cnn_mnist.py`、`examples/cnn_mnist_benchmark.py`、`tests/test_cnn_mnist.py`、README command、ignored local metrics file `runs/smoke/cnn_mnist_metrics.json`。
- Planned next step：real MNIST/Fashion-MNIST local-file loader、curated local result、TPU run、local-vs-TPU comparison。

### Demo 2: Existing JAX Model or Training Stack

- Topic：使用 JAX ecosystem 中較高階的 training approach，例如 Flax + Optax，或小型既有 JAX model/example。
- Goal：和 Demo 1 的 raw-JAX implementation 對照，觀察 framework abstraction 帶來的結構差異。
- Emphasis：
  - Flax/Optax 提供哪些 model definition、parameter handling、optimizer abstraction。
  - 訓練程式碼是否更短、更容易維護或更容易擴充。
  - metrics 與 benchmark artifacts 是否能沿用 Demo 1 的 JSON format。
  - local vs TPU workflow 是否能重用 Demo 1 的 setup、run、collect、compare 流程。
- Current status：planned, not implemented。
- Evidence now：目前沒有 Demo 2 code、results 或 tests。
- Planned next step：先選定最小 Flax + Optax 或既有 JAX example，保持 CPU-friendly defaults，避免引入過重 dependencies。

### Demo 3: Gemma or Pretrained-Model Cloud Demo

- Topic：在範圍、授權、資源與時間可控時，展示 Gemma-like 或其他 pretrained-model workflow on Google Cloud / TPU-related environment。
- Goal：說明 pretrained model workflow 和小型 hand-written / standard training demo 的差異。
- Emphasis：
  - Model source、license、access requirements。
  - Dependency group and setup requirements。
  - Expected hardware and memory requirements。
  - Demo 是 inference-only、fine-tuning，或只是 workflow demonstration。
  - 若 model access、quota、hardware 或時間不足，如何退回文件化流程或小型替代展示。
- Current status：optional advanced planned work, not implemented。
- Evidence now：目前沒有 Gemma code、model downloads、model access evidence、TPU runs 或 results。
- Planned next step：等 Demo 1 cloud workflow 穩定後，再評估是否納入 final scope。

## Slide-by-Slide Outline

### Slide 1: Project Direction

- Key message：本專題以 JAX training workflow 與 Google Cloud TPU benchmark 為主軸。
- Content bullets：
  - Course project: Numerical Computation with JAX
  - Main direction: Hand-written CNN on MNIST / Fashion-MNIST
  - Longer roadmap: raw JAX demo, higher-level JAX stack demo, optional pretrained-model cloud demo
  - Current stage: local Demo 1 foundation
- Demo or evidence to show：README opening and Demo 1 section。
- Estimated time：0:45
- Current status：completed for project direction documentation; implementation partial

### Slide 2: Three-Demo Roadmap

- Key message：專案會用三個 demo 逐步展示從 raw JAX 到 higher-level stack，再到 optional pretrained workflow。
- Content bullets：
  - Demo 1: raw-JAX hand-written CNN benchmark
  - Demo 2: Flax + Optax or existing JAX model/training stack comparison
  - Demo 3: optional Gemma-like or pretrained-model cloud workflow
  - Only Demo 1 local foundation exists now
- Demo or evidence to show：roadmap table with completed / partial / planned labels。
- Estimated time：1:00
- Current status：Demo 1 partial; Demo 2 planned; Demo 3 optional planned

### Slide 3: Repository and Tooling Foundation

- Key message：local development foundation 已建立，讓後續 demo 可以用同一套 basic validation workflow。
- Content bullets：
  - `src/` layout，package name: `jax_tpu_project`
  - `uv` setup with `pyproject.toml` and `uv.lock`
  - Ruff and pytest configured in `pyproject.toml`
  - Runtime check: `scripts/check_jax_device.sh` and `jax_tpu_project.cli devices`
  - `.gitignore` excludes generated runs, credentials, raw data, model weights
- Demo or evidence to show：`pyproject.toml`、`scripts/check_jax_device.sh`、`.gitignore`。
- Estimated time：0:55
- Current status：completed

### Slide 4: Demo 1 Goal and Current Scope

- Key message：Demo 1 的已完成部分是 local raw-JAX CNN benchmark foundation，不是完整 MNIST/Fashion-MNIST 或 TPU result。
- Content bullets：
  - Goal: CNN on MNIST / Fashion-MNIST local vs TPU benchmark
  - Current model: hand-written raw JAX CNN
  - Current data: synthetic MNIST-shaped data
  - Current output: JSON benchmark metrics
  - Planned: real dataset loader and TPU execution
- Demo or evidence to show：README Demo 1 command and `SUPPORTED_DATASETS` behavior。
- Estimated time：0:55
- Current status：partial

### Slide 5: Demo 1 CNN Architecture

- Key message：CNN architecture intentionally stays small and explainable for course presentation and benchmark workflow setup。
- Content bullets：
  - Input shape: `[batch, 28, 28, 1]`
  - Conv1: 3x3, 1 to 8 channels
  - Average pooling by reshape + mean
  - Conv2: 3x3, 8 to 16 channels
  - Dense hidden: `7 * 7 * 16` to 32
  - Logits: 32 to 10 classes
- Demo or evidence to show：`init_cnn_params()` and `forward()` in `src/jax_tpu_project/cnn_mnist.py`。
- Estimated time：1:10
- Current status：completed for synthetic benchmark foundation

### Slide 6: Demo 1 JAX Training Step

- Key message：training step 展示 raw JAX 的核心 building blocks，後續可作為 TPU benchmark 的共同 workload。
- Content bullets：
  - `jax.jit` on `training_step`
  - `jax.value_and_grad` for loss and gradients
  - explicit SGD update using `jax.tree_util.tree_map`
  - `jax.lax.conv_general_dilated` for convolution
  - `jax.vmap` in synthetic data template generation
  - `block_until_ready()` for timing correctness
- Demo or evidence to show：`training_step()`、`make_synthetic_mnist_data()`、`_block_training_outputs()`。
- Estimated time：1:10
- Current status：completed

### Slide 7: Dataset and Benchmark Evidence

- Key message：目前的數據證據只支持 local synthetic smoke benchmark。
- Content bullets：
  - Implemented path: `synthetic`
  - `mnist` and `fashion_mnist` are CLI choices but reserved for future local-file support
  - Existing ignored artifact: `runs/smoke/cnn_mnist_metrics.json`
  - Artifact records CPU backend, 3 steps, 1 warmup step, batch size 16, seed 0
  - This is workflow evidence, not final benchmark conclusion
- Demo or evidence to show：open metrics JSON and point to backend, devices, timing, loss, accuracy fields。
- Estimated time：1:15
- Current status：completed for local synthetic smoke; real datasets planned

### Slide 8: Demo 1 Live Demo

- Key message：目前可現場展示的是 local synthetic benchmark and metrics generation。
- Content bullets：
  - Run JAX device summary
  - Run short benchmark with synthetic data
  - Show generated JSON metrics
  - Show tests that protect benchmark behavior
- Demo or evidence to show：
  ```bash
  uv run python examples/cnn_mnist_benchmark.py \
    --dataset synthetic \
    --steps 3 \
    --batch-size 16 \
    --seed 0 \
    --output-dir runs/smoke \
    --platform-label local
  ```
- Estimated time：1:30
- Current status：completed for local smoke demo

### Slide 9: Test Coverage

- Key message：tests are lightweight and aligned with current local smoke scope。
- Content bullets：
  - Runtime summary JSON serialization and fields
  - CNN parameter shapes
  - Forward logits shape
  - Training smoke behavior
  - Metrics writing
  - Deterministic seed behavior
  - No tests yet for TPU, real MNIST, Fashion-MNIST, Demo 2, or Demo 3
- Demo or evidence to show：`tests/test_runtime.py` and `tests/test_cnn_mnist.py`。
- Estimated time：0:55
- Current status：completed for current scope

### Slide 10: Demo 1 TPU Benchmark Plan

- Key message：TPU benchmark remains planned work and will need reproducible cloud evidence。
- Content bullets：
  - Prepare TPU VM setup and cleanup documentation
  - Run same benchmark with `platform-label=tpu`
  - Capture JSON metrics, terminal logs, backend/devices, monitoring evidence
  - Compare local and TPU with matching benchmark configuration where practical
  - Document cost, quota, availability, and interpretation limits
- Demo or evidence to show：planned workflow checklist, not completed result。
- Estimated time：1:00
- Current status：planned

### Slide 11: Demo 2 and Demo 3 Planned Extensions

- Key message：later demos broaden the project from raw JAX to ecosystem workflow and optional pretrained-model workflow。
- Content bullets：
  - Demo 2: Flax + Optax or existing JAX example; compare abstraction and code complexity
  - Demo 2 should reuse metrics format and local/TPU workflow if practical
  - Demo 3: optional Gemma-like/pretrained cloud demo
  - Demo 3 depends on model access, license, dependencies, hardware, memory, quota, and time
  - No code or results exist yet for either demo
- Demo or evidence to show：roadmap status table with planned labels。
- Estimated time：1:10
- Current status：planned / optional planned

### Slide 12: Risks, Limitations, and Next Steps

- Key message：目前可以展示完整 local foundation，但最終 benchmark conclusions 還需要更多 evidence。
- Content bullets：
  - Synthetic data is not a real MNIST/Fashion-MNIST result
  - 3-step smoke run is not a stable performance benchmark
  - TPU execution, monitoring, and comparison are not completed
  - Demo 2 and Demo 3 must stay scoped to avoid overexpansion
  - Next: dataset loader, curated local result, TPU workflow, Demo 2 selection, optional Demo 3 feasibility check
- Demo or evidence to show：completed vs planned checklist。
- Estimated time：1:10
- Current status：partial / planned

## Suggested Live Demo Flow

1. Show repository entry point:
   ```bash
   sed -n '1,160p' README.md
   ```
2. Show JAX runtime device summary:
   ```bash
   uv run python -m jax_tpu_project.cli devices
   ```
3. Run short local benchmark:
   ```bash
   uv run python examples/cnn_mnist_benchmark.py \
     --dataset synthetic \
     --steps 3 \
     --batch-size 16 \
     --seed 0 \
     --output-dir runs/smoke \
     --platform-label local
   ```
4. Show generated metrics:
   ```bash
   sed -n '1,220p' runs/smoke/cnn_mnist_metrics.json
   ```
5. Show tests that protect the current scope:
   ```bash
   sed -n '1,220p' tests/test_cnn_mnist.py
   ```
6. Close with roadmap slide for Demo 1 TPU work, Demo 2, and optional Demo 3.

## Fallback Plan If Live Demo or TPU Access Fails

- If local demo fails because environment is not ready, show README commands, `pyproject.toml`, tests, and expected output fields.
- If JAX backend/device output differs from prior local artifact, explain that backend/device metadata is part of benchmark evidence.
- If TPU access is unavailable, explicitly state TPU execution is planned work and present the Demo 1 cloud checklist.
- If dataset or network access is unavailable, use synthetic mode because it is intentionally offline and deterministic.
- If Demo 2 is not ready, keep it as a comparison plan rather than implying implementation exists.
- If Demo 3 access, quota, hardware, or time is unavailable, present it as optional advanced work with a documented fallback.
- If metrics file is missing, rerun the smoke command locally; do not fabricate metrics.

## What Results Can Honestly Be Claimed Now

- The repository has an initial Python/JAX project scaffold using `uv`, `pyproject.toml`, and `src/jax_tpu_project/`.
- The repository includes Ruff and pytest configuration.
- The repository includes a JAX runtime/device summary CLI and shell helper.
- Demo 1 implements a hand-written raw-JAX CNN benchmark foundation for MNIST-shaped images.
- The current completed data path is deterministic synthetic MNIST-shaped data.
- The benchmark writes JSON metrics with backend, devices, dataset, seed, batch size, timing, throughput, loss, accuracy, and output path.
- Tests cover runtime summary behavior, CNN parameter shapes, forward pass shape, training smoke behavior, metrics writing, and deterministic seed behavior.
- A local ignored smoke metrics file exists under `runs/smoke/cnn_mnist_metrics.json`, showing a CPU synthetic run with 3 timed steps, batch size 16, seed 0, and warmup steps 1.

## What Results Are Still Planned

- Real MNIST local-file loader.
- Fashion-MNIST local-file loader.
- Longer local benchmark runs with curated result artifacts under `report/results/`.
- Google Cloud TPU setup and execution workflow.
- Actual TPU benchmark run with evidence.
- Cloud monitoring notes or screenshots.
- Local-vs-TPU comparison table or plot.
- Demo 2 implementation using Flax + Optax or an existing JAX model/example.
- Demo 2 tests, metrics, and comparison against Demo 1.
- Optional Demo 3 pretrained-model cloud workflow, including access/license/dependency/hardware review.
- Final interpretation of speed, throughput, cost, limitations, and backend differences.

## Risks and Limitations

- Synthetic data is useful for workflow validation but not a valid real dataset result.
- Current local metrics are generated under ignored `runs/`; they should be curated before being treated as report evidence.
- A 3-step smoke run is not enough for stable performance conclusions.
- TPU execution has not been performed, so no cloud performance claim should be made yet.
- Small CNN workloads may not saturate TPU hardware; comparison must separate compilation, warmup, and steady-state timing.
- Demo 2 may introduce dependencies and abstractions that need careful scoping.
- Demo 3 may be blocked by model license, access approval, quota, memory requirements, or setup time.

## Next Steps

1. Implement local-file MNIST and Fashion-MNIST loaders without network dependency in default tests.
2. Add small curated local Demo 1 benchmark artifacts under `report/results/` with commands and configuration.
3. Draft Demo 1 cloud TPU setup, execution, monitoring, retrieval, and cleanup documentation once local dataset workflow is stable.
4. Run Demo 1 on TPU VM and capture JSON metrics, terminal logs, backend/devices, and monitoring evidence.
5. Generate a local-vs-TPU comparison table or figure from saved Demo 1 metrics.
6. Choose a minimal Demo 2 approach, such as Flax + Optax, and design it to reuse metrics and benchmark artifact conventions.
7. Evaluate Demo 3 feasibility only after Demo 1 cloud workflow is stable: model source, license/access, dependencies, hardware, memory, quota, and fallback scope.

## Possible Final-Report Structure

1. 專案動機與三個 demo roadmap
2. JAX runtime and backend/device inspection
3. Demo 1: raw-JAX CNN implementation
4. Demo 1 dataset strategy: synthetic smoke data, MNIST, Fashion-MNIST
5. Demo 1 local benchmark methodology and metrics
6. Demo 1 Google Cloud TPU setup and execution workflow
7. Demo 1 local vs TPU results and comparison
8. Demo 2: higher-level JAX training stack comparison
9. Demo 3: optional pretrained-model cloud workflow and feasibility notes
10. Monitoring, reproducibility evidence, and artifact management
11. Limitations, failed attempts, and lessons learned
12. Future work
