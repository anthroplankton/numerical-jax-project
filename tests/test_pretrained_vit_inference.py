from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "pretrained_vit_inference.py"
SAMPLE_IMAGE_PATH = Path("examples/assets/chihuahua_pet_licorice.jpg")


def load_vit_example_module():
    spec = importlib.util.spec_from_file_location(
        "pretrained_vit_inference", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_safe_defaults() -> None:
    module = load_vit_example_module()

    # These tests verify argparse behavior only; they do not read the image file.
    args = module.parse_args(["--image", str(SAMPLE_IMAGE_PATH)])

    assert args.model == "google/vit-base-patch16-224"
    assert args.image == SAMPLE_IMAGE_PATH
    assert args.batch_size == 1
    assert args.warmup_steps == 1
    assert args.benchmark_steps == 5
    assert args.jax_platform == "cpu"
    assert args.output == Path("runs/vit-inference/metrics.json")


def test_parse_args_accepts_benchmark_settings() -> None:
    module = load_vit_example_module()

    args = module.parse_args(
        [
            "--image",
            str(SAMPLE_IMAGE_PATH),
            "--output",
            "runs/test/metrics.json",
            "--batch-size",
            "4",
            "--warmup-steps",
            "0",
            "--benchmark-steps",
            "2",
            "--jax-platform",
            "default",
        ]
    )

    assert args.image == SAMPLE_IMAGE_PATH
    assert args.output == Path("runs/test/metrics.json")
    assert args.batch_size == 4
    assert args.warmup_steps == 0
    assert args.benchmark_steps == 2
    assert args.jax_platform == "default"


def test_parse_args_accepts_jax_platform_choices() -> None:
    module = load_vit_example_module()

    for platform in ("default", "cpu", "cuda", "tpu"):
        args = module.parse_args(
            ["--image", str(SAMPLE_IMAGE_PATH), "--jax-platform", platform]
        )

        assert args.image == SAMPLE_IMAGE_PATH
        assert args.jax_platform == platform


def test_parse_args_rejects_unknown_jax_platform() -> None:
    module = load_vit_example_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--image", str(SAMPLE_IMAGE_PATH), "--jax-platform", "gpu"])


def test_parse_args_rejects_invalid_step_counts() -> None:
    module = load_vit_example_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--image", str(SAMPLE_IMAGE_PATH), "--batch-size", "0"])

    with pytest.raises(SystemExit):
        module.parse_args(["--image", str(SAMPLE_IMAGE_PATH), "--warmup-steps", "-1"])

    with pytest.raises(SystemExit):
        module.parse_args(["--image", str(SAMPLE_IMAGE_PATH), "--benchmark-steps", "0"])


def test_build_metrics_computes_throughput_and_preserves_schema() -> None:
    module = load_vit_example_module()

    metrics = module.build_metrics(
        model_name="google/vit-base-patch16-224",
        selected_jax_platform="cpu",
        backend="cpu",
        devices=[
            {
                "platform": "cpu",
                "device_kind": "cpu",
                "id": 0,
                "repr": "TFRT_CPU_0",
            }
        ],
        input_shape=(2, 3, 224, 224),
        batch_size=2,
        warmup_steps=1,
        benchmark_steps=2,
        step_times=[0.2, 0.4],
        predicted_index=285,
        predicted_label="Egyptian cat",
    )

    assert metrics == {
        "model_name": "google/vit-base-patch16-224",
        "selected_jax_platform": "cpu",
        "backend": "cpu",
        "devices": [
            {
                "platform": "cpu",
                "device_kind": "cpu",
                "id": 0,
                "repr": "TFRT_CPU_0",
            }
        ],
        "input_shape": [2, 3, 224, 224],
        "batch_size": 2,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "mean_step_time_sec": pytest.approx(0.3),
        "throughput_images_per_sec": pytest.approx(2 / 0.3),
        "predicted_index": 285,
        "predicted_label": "Egyptian cat",
    }


def test_lookup_label_accepts_string_or_integer_keys() -> None:
    module = load_vit_example_module()

    assert module.lookup_label({1: "one"}, 1) == "one"
    assert module.lookup_label({"2": "two"}, 2) == "two"
    assert module.lookup_label({}, 3) == "3"


def test_apply_jax_platform_sets_environment_before_runtime_import(monkeypatch) -> None:
    module = load_vit_example_module()
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    module.apply_jax_platform("cpu")

    assert module.os.environ["JAX_PLATFORMS"] == "cpu"


def test_apply_jax_platform_default_leaves_environment_unchanged(monkeypatch) -> None:
    module = load_vit_example_module()
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")

    module.apply_jax_platform("default")

    assert module.os.environ["JAX_PLATFORMS"] == "cuda"


def test_apply_jax_platform_rejects_invalid_value() -> None:
    module = load_vit_example_module()

    with pytest.raises(ValueError):
        module.apply_jax_platform("gpu")
