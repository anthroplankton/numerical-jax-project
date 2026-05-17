# Current Project Status

## Current Presentation Scope

因為課程時間與展示限制，目前簡報與 demo 主軸收斂到 **Demo 2:
pretrained ViT inference benchmark with JAX/Flax**。

目前主要展示路徑：

- model：`google/vit-base-patch16-224`
- runtime：JAX/Flax
- demo script：`examples/pretrained_vit_inference.py`
- stable classroom backend：local CPU
- sample image：`examples/assets/chihuahua_pet_licorice.jpg`
- curated local CPU result artifacts：`report/results/`
- next major goal：prepare and run Google Cloud TPU workflow for Demo 2

TPU execution has not been attempted or completed yet. No TPU performance claim
should be made until a real TPU VM run, metrics, logs, monitoring notes, and
cleanup evidence exist.

## What The Repository Currently Does

已完成的共用 project foundation：

- Python package 使用 `src/jax_tpu_project/` layout。
- Project setup 使用 `uv`、`pyproject.toml`、`uv.lock`。
- Ruff and pytest 設定已放在 `pyproject.toml`。
- JAX runtime/device sanity check 已實作：
  - `src/jax_tpu_project/runtime.py`
  - `src/jax_tpu_project/cli.py`
  - `scripts/check_jax_device.sh`

已完成的 Demo 2 重點：

- Optional `pretrained` dependency group for Transformers/Flax/Pillow/Hugging
  Face workflow。
- Pretrained ViT inference script:
  - `examples/pretrained_vit_inference.py`
- Local CPU classroom sample image:
  - `examples/assets/chihuahua_pet_licorice.jpg`
  - `examples/assets/README.md`
- Optional private live-demo image set support:
  - recommended local-only path: `data/local/demo2_vit_images/`
  - optional manifest: `data/local/demo2_vit_images/manifest.txt`
  - manifest mode uses real mixed-image batches
  - the final partial batch is padded by repeating its last real image; padded
    entries are ignored for predictions and throughput, and `num_padded_images`
    records the padding count
  - qualitative live predictions only; not a public dataset or accuracy benchmark
- Demo 2 documentation:
  - `docs/pretrained_vit_demo.md`
- Planned TPU workflow documentation:
  - `cloud/demo2_vit_tpu_workflow.md`
- Lightweight tests that do not download model weights:
  - `tests/test_pretrained_vit_inference.py`

## Demo 2 Local CPU Evidence

Manual local checks completed:

- Hugging Face model download succeeded for `google/vit-base-patch16-224`。
- Local CPU inference succeeded。
- Prediction for the sample image was `Chihuahua`。
- Curated JSON artifacts:
  - `report/results/demo2_vit_local_cpu_b1.json`
  - `report/results/demo2_vit_local_cpu_b4.json`
  - `report/results/demo2_vit_local_cpu_b8.json`
  - `report/results/README.md`

Observed local CPU throughput:

| Artifact | Batch size | Mean step time | Throughput |
| --- | ---: | ---: | ---: |
| `demo2_vit_local_cpu_b1.json` | 1 | 0.18744530999993003 s | 5.334889413879565 images/s |
| `demo2_vit_local_cpu_b4.json` | 4 | 0.8838171279999187 s | 4.5258231293299485 images/s |
| `demo2_vit_local_cpu_b8.json` | 8 | 1.973294752699894 s | 4.054133316401044 images/s |

These are single-image repeated-batch inference results. They are not
dataset-level accuracy evaluation, not GPU results, and not TPU results.
Private manifest runs follow the same qualitative-inference framing unless
explicit labels and top-k evaluation are added later, but they now measure true
batched manifest inference over the listed images.

## Local CUDA Limitation

On the laptop used for local checks, simple JAX GPU matrix multiplication
worked, but the ViT-like convolution path failed during cuDNN autotuning.
Therefore local CUDA is documented as a limitation and is not used as Demo 2
benchmark evidence.

## Demo 1 Status

Demo 1 remains in the repository as preserved background/foundation work.

Implemented:

- raw-JAX hand-written CNN benchmark foundation:
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
  - `tests/test_cnn_mnist.py`
- deterministic synthetic MNIST-shaped data path
- JSON metrics output for local smoke runs

Not completed:

- real MNIST/Fashion-MNIST local-file loader
- curated Demo 1 result under `report/results/`
- Demo 1 TPU execution and local-vs-TPU comparison

Demo 1 is not the current presentation focus.

## Demo 3 Status

Demo 3 remains optional future work for a larger pretrained-model or Gemma-like
cloud workflow. It is not the current presentation focus and has no completed
code, model access notes, cloud run, TPU run, or result artifacts.

## Current Validation Scope

Default tests remain lightweight and offline:

- `tests/test_runtime.py`
- `tests/test_cnn_mnist.py`
- `tests/test_pretrained_vit_inference.py`

The Demo 2 tests check argument parsing, metrics helper behavior, and platform
environment helper behavior. They do not require GPU, TPU, network access,
Hugging Face access, image opening, or model weight download.

## Next Technical Milestones

1. Review `cloud/demo2_vit_tpu_workflow.md` and confirm project, zone, TPU
   accelerator type, quota, and cost constraints.
2. Run JAX backend/device verification on a TPU VM.
3. Run Demo 2 on TPU with `--jax-platform tpu`.
4. Save TPU JSON metrics, logs, monitoring notes, and cleanup evidence.
5. Compare Demo 2 local CPU and TPU results using curated artifacts.
6. Keep Demo 1 and Demo 3 preserved as future work unless course scope changes.
