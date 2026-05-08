# Current Project Status

## What the Project Currently Does

目前 repository 是一個小型 JAX course project scaffold，核心方向是「Hand-written CNN on MNIST / Fashion-MNIST, designed for local vs Google Cloud TPU training benchmark」。

已完成的實作重點：

- Python package 使用 `src/jax_tpu_project/` layout。
- Project setup 使用 `uv`、`pyproject.toml`、`uv.lock`。
- Ruff and pytest 設定已放在 `pyproject.toml`。
- JAX runtime/device sanity check 已實作：
  - `src/jax_tpu_project/runtime.py`
  - `src/jax_tpu_project/cli.py`
  - `scripts/check_jax_device.sh`
- Demo 1 已建立 raw-JAX hand-written CNN benchmark foundation：
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
- 目前 benchmark 使用 deterministic synthetic MNIST-shaped data，shape 為 `[N, 28, 28, 1]`，labels 為 0 到 9。
- Benchmark 會輸出 JSON metrics，供後續 local vs TPU comparison 使用。

## Planned Demo Roadmap

### Demo 1: Hand-Written Model

- Purpose：用 raw JAX 實作 CNN on MNIST / Fashion-MNIST，作為 local vs Google Cloud TPU training benchmark 的主要 workload。
- Current status：local raw-JAX CNN benchmark foundation exists；real dataset and TPU comparison are not completed。
- Evidence exists now：
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
  - `tests/test_cnn_mnist.py`
  - README benchmark commands
  - ignored local generated metrics: `runs/smoke/cnn_mnist_metrics.json`
- Not implemented yet：
  - Real MNIST local-file loader
  - Fashion-MNIST local-file loader
  - Curated local benchmark result under `report/results/`
  - TPU run and local-vs-TPU comparison
- Likely next implementation step：add local-file dataset loading for MNIST/Fashion-MNIST, then produce a curated local benchmark artifact before TPU execution。

### Demo 2: Existing JAX Model or Training Stack

- Purpose：使用 Flax + Optax 或小型既有 JAX model/example，和 Demo 1 的 raw-JAX implementation 做 workflow 對照。
- Current status：planned, not implemented。
- Evidence exists now：none；目前沒有 Demo 2 code、tests、metrics 或 results。
- Not implemented yet：
  - Model/training stack selection
  - Dependency decision
  - Training script or example
  - Shared metrics format
  - Local smoke tests
  - TPU workflow reuse
- Likely next implementation step：選一個最小且可說明 abstraction tradeoff 的 Flax + Optax 或 existing JAX example，先規劃如何沿用 Demo 1 的 JSON metrics format。

### Demo 3: Gemma or Pretrained-Model Cloud Demo

- Purpose：在 scope、access、quota、hardware、time 都可控時，展示 Gemma-like 或其他 pretrained-model workflow 與小型訓練 demo 的差異。
- Current status：optional advanced planned work, not implemented。
- Evidence exists now：none；目前沒有 Gemma code、model downloads、model access evidence、cloud run、TPU run 或 results。
- Not implemented yet：
  - Model source and license/access review
  - Dependency group and setup plan
  - Hardware and memory requirement estimate
  - Decision between inference-only, fine-tuning, or workflow demonstration
  - Fallback plan for unavailable access/quota/hardware/time
- Likely next implementation step：等 Demo 1 cloud workflow 穩定後，再做 feasibility check，決定是否納入 final presentation/report。

## How to Run the Current Local Demo

先確認 JAX runtime/device：

```bash
bash scripts/check_jax_device.sh
```

或直接執行 package CLI：

```bash
uv run python -m jax_tpu_project.cli devices
```

執行 quick local smoke benchmark：

```bash
uv run python examples/cnn_mnist_benchmark.py \
  --dataset synthetic \
  --steps 3 \
  --batch-size 16 \
  --seed 0 \
  --output-dir runs/smoke \
  --platform-label local
```

執行稍長的 local benchmark：

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

## Output and Metrics

Benchmark terminal output 會包含：

- per-step loss
- synthetic accuracy
- step time
- active JAX backend
- visible devices
- average step time
- examples per second
- metrics file path

每次 run 會寫出：

```text
<output-dir>/cnn_mnist_metrics.json
```

JSON metrics 目前包含：

- `backend`
- `devices`
- `platform_label`
- `dataset_name`
- `seed`
- `batch_size`
- `steps`
- `warmup_steps`
- `learning_rate`
- `total_training_time_seconds`
- `average_step_time_seconds`
- `examples_per_second`
- `initial_loss`
- `final_loss`
- `initial_accuracy`
- `final_accuracy`
- `output_artifact_path`

目前 repository 工作區中有一個 ignored local generated artifact：

```text
runs/smoke/cnn_mnist_metrics.json
```

這個 artifact 顯示一次 local CPU synthetic smoke run：

- dataset: `synthetic`
- backend: `cpu`
- device: `cpu:0`
- platform label: `local`
- steps: 3
- warmup steps: 1
- batch size: 16
- seed: 0

因為 `runs/` 被 `.gitignore` 排除，這個檔案應視為 local generated evidence，而不是已整理好的 final report result。

## What Has Been Tested

目前 tests 包含：

- `tests/test_runtime.py`
  - `get_device_summary()` 可以 JSON serialize。
  - device summary 包含 `default_backend` and visible device fields。
- `tests/test_cnn_mnist.py`
  - CNN parameter initialization shapes。
  - Forward pass returns logits with shape `[batch, 10]`。
  - Two training steps can run and produce finite losses。
  - Benchmark writes `cnn_mnist_metrics.json` with expected metadata。
  - Fixed seed produces deterministic synthetic images/labels and close metrics。

這些 tests 是 local CPU-friendly smoke checks，符合目前 scaffold and synthetic benchmark scope。

## What Has Not Been Tested

尚未測試或沒有 repository evidence 的項目：

- Google Cloud TPU execution。
- TPU backend/device detection in this project。
- TPU training metrics。
- Cloud monitoring, logs, screenshots, or dashboards。
- Local vs TPU comparison。
- Real MNIST training。
- Fashion-MNIST training。
- Longer benchmark stability or repeated-run statistics。
- Demo 2 existing JAX stack workflow。
- Demo 3 Gemma/pretrained-model workflow。
- Docker, CI, notebooks, pretrained model workflows。

## What Is Not Yet Implemented

尚未完成的功能：

- Real MNIST local-file loader。
- Fashion-MNIST local-file loader。
- Dataset download workflow。
- Curated benchmark results under `report/results/`。
- Plotting or comparison scripts。
- Google Cloud TPU setup, execution, monitoring, and cleanup documentation。
- TPU execution command and result collection workflow。
- Final local-vs-TPU comparison table or figure。
- Demo 2 model/training stack, tests, metrics, and comparison。
- Demo 3 model access, setup, run command, fallback materials, and results。

重要狀態說明：

- `SUPPORTED_DATASETS` includes `synthetic`, `mnist`, and `fashion_mnist`。
- 但目前只有 `synthetic` path 實作完成。
- 選擇 `mnist` or `fashion_mnist` 目前會 raise `FileNotFoundError`，訊息表示這些 dataset reserved for future local-file support。

## Suggested Next Technical Milestones

1. Add real MNIST and Fashion-MNIST local-file loading without making network access required for default tests.
2. Add a small curated Demo 1 local benchmark result under `report/results/` with command, environment, and metrics.
3. Decide Demo 1 benchmark configurations for local and TPU runs: batch size, steps, warmup steps, seed, learning rate.
4. Draft Google Cloud TPU workflow documentation for Demo 1: setup, code transfer, environment install, run command, monitoring, output retrieval, cleanup.
5. Run Demo 1 on TPU and save JSON metrics plus terminal/cloud evidence.
6. Build a local-vs-TPU comparison table or figure from saved Demo 1 metrics.
7. Choose the smallest useful Demo 2 JAX ecosystem approach and design shared metrics output.
8. Evaluate whether Demo 3 is feasible after Demo 1 cloud workflow is stable.
9. Update final report with limitations, failed attempts, and interpretation of results.
