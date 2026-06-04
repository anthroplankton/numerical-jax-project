from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "pretrained_vit_inference.py"
SAMPLE_IMAGE_PATH = Path("examples/assets/chihuahua_pet_licorice.jpg")
PUBLIC_MANIFEST_PATH = Path("examples/assets/manifest.txt")
PRIVATE_MANIFEST_PATH = Path("data/local/demo2_vit_images/manifest.txt")


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
    assert args.top_k == 5
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
            "--top-k",
            "3",
        ]
    )

    assert args.image == SAMPLE_IMAGE_PATH
    assert args.output == Path("runs/test/metrics.json")
    assert args.batch_size == 4
    assert args.warmup_steps == 0
    assert args.benchmark_steps == 2
    assert args.jax_platform == "default"
    assert args.top_k == 3


def test_parse_args_accepts_private_image_manifest() -> None:
    module = load_vit_example_module()

    args = module.parse_args(["--image-manifest", str(PRIVATE_MANIFEST_PATH)])

    assert args.image is None
    assert args.image_manifest == PRIVATE_MANIFEST_PATH


def test_parse_args_rejects_image_and_manifest_together() -> None:
    module = load_vit_example_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--image",
                str(SAMPLE_IMAGE_PATH),
                "--image-manifest",
                str(PRIVATE_MANIFEST_PATH),
            ]
        )


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


def test_read_image_manifest_resolves_paths_without_reading_images(tmp_path) -> None:
    module = load_vit_example_module()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "# private live-demo images",
                "demo2_private_001.jpg",
                "nested/demo2_private_002.png",
                str(SAMPLE_IMAGE_PATH),
                "",
            ]
        )
    )

    image_paths = module.read_image_manifest(manifest)

    assert image_paths == [
        tmp_path / "demo2_private_001.jpg",
        tmp_path / "nested" / "demo2_private_002.png",
        tmp_path / SAMPLE_IMAGE_PATH,
    ]


def test_read_image_manifest_rejects_empty_manifest(tmp_path) -> None:
    module = load_vit_example_module()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# no images yet\n\n")

    with pytest.raises(ValueError, match="contains no image paths"):
        module.read_image_manifest(manifest)


def test_resolve_image_paths_prefers_single_image_over_manifest() -> None:
    module = load_vit_example_module()

    image_paths = module.resolve_image_paths(SAMPLE_IMAGE_PATH, PRIVATE_MANIFEST_PATH)

    assert image_paths == [SAMPLE_IMAGE_PATH]


def test_private_image_manifest_example_path_uses_ignored_data_local() -> None:
    module = load_vit_example_module()

    assert module.private_image_manifest_example_path() == PRIVATE_MANIFEST_PATH


def test_classify_manifest_path_marks_public_and_local_manifests() -> None:
    module = load_vit_example_module()

    assert module.classify_manifest_path(PUBLIC_MANIFEST_PATH) == "public_example"
    assert module.classify_manifest_path(PRIVATE_MANIFEST_PATH) == "local_private"
    assert (
        module.classify_manifest_path(Path("datasets/manifest.txt")) == "local_manifest"
    )


def test_public_example_manifest_lists_expected_images_without_opening_files() -> None:
    module = load_vit_example_module()

    image_paths = module.read_image_manifest(PUBLIC_MANIFEST_PATH)

    assert image_paths == [
        Path("examples/assets/chihuahua_pet_licorice.jpg"),
        Path("examples/assets/adelie_penguins_brooding.jpg"),
        Path("examples/assets/doge_homemade_meme.jpg"),
        Path("examples/assets/polar_bear_zoo_face.jpg"),
        Path("examples/assets/black_cat_staring_closeup.jpg"),
    ]


def test_build_manifest_batch_specs_matches_public_example_counts() -> None:
    module = load_vit_example_module()

    image_count = len(module.read_image_manifest(PUBLIC_MANIFEST_PATH))

    assert image_count == 5
    assert module.build_manifest_batch_specs(num_images=image_count, batch_size=1) == [
        {"start_index": 0, "end_index": 1, "real_size": 1, "padding_count": 0},
        {"start_index": 1, "end_index": 2, "real_size": 1, "padding_count": 0},
        {"start_index": 2, "end_index": 3, "real_size": 1, "padding_count": 0},
        {"start_index": 3, "end_index": 4, "real_size": 1, "padding_count": 0},
        {"start_index": 4, "end_index": 5, "real_size": 1, "padding_count": 0},
    ]

    assert module.build_manifest_batch_specs(num_images=image_count, batch_size=4) == [
        {
            "start_index": 0,
            "end_index": 4,
            "real_size": 4,
            "padding_count": 0,
        },
        {
            "start_index": 4,
            "end_index": 5,
            "real_size": 1,
            "padding_count": 3,
        },
    ]


def test_build_manifest_batch_specs_matches_local_live_demo_counts() -> None:
    module = load_vit_example_module()

    local_live_demo_image_count = 15

    assert module.build_manifest_batch_specs(
        num_images=local_live_demo_image_count, batch_size=1
    )[-1] == {
        "start_index": 14,
        "end_index": 15,
        "real_size": 1,
        "padding_count": 0,
    }

    batch_specs = module.build_manifest_batch_specs(
        num_images=local_live_demo_image_count, batch_size=4
    )

    assert batch_specs == [
        {
            "start_index": 0,
            "end_index": 4,
            "real_size": 4,
            "padding_count": 0,
        },
        {
            "start_index": 4,
            "end_index": 8,
            "real_size": 4,
            "padding_count": 0,
        },
        {
            "start_index": 8,
            "end_index": 12,
            "real_size": 4,
            "padding_count": 0,
        },
        {
            "start_index": 12,
            "end_index": 15,
            "real_size": 3,
            "padding_count": 1,
        },
    ]


def test_format_top_k_predictions_preserves_indices_labels_and_scores() -> None:
    module = load_vit_example_module()

    prediction = module.format_top_k_predictions(
        indices=[285, 207, 208],
        scores=[0.7, 0.2, 0.1],
        id2label={285: "Egyptian cat", "207": "golden retriever", 208: "Labrador"},
    )

    assert prediction == {
        "top1_index": 285,
        "top1_label": "Egyptian cat",
        "top5": [
            {"index": 285, "label": "Egyptian cat", "score": pytest.approx(0.7)},
            {
                "index": 207,
                "label": "golden retriever",
                "score": pytest.approx(0.2),
            },
            {"index": 208, "label": "Labrador", "score": pytest.approx(0.1)},
        ],
    }


def test_build_image_metrics_computes_per_image_throughput() -> None:
    module = load_vit_example_module()

    metrics = module.build_image_metrics(
        image_path=SAMPLE_IMAGE_PATH,
        input_shape=(2, 3, 224, 224),
        batch_size=2,
        benchmark_steps=2,
        step_times=[0.2, 0.4],
        prediction={
            "top1_index": 285,
            "top1_label": "Egyptian cat",
            "top5": [
                {
                    "index": 285,
                    "label": "Egyptian cat",
                    "score": 0.9,
                }
            ],
        },
    )

    assert metrics == {
        "image_path": str(SAMPLE_IMAGE_PATH),
        "input_shape": [2, 3, 224, 224],
        "batch_size": 2,
        "benchmark_steps": 2,
        "mean_step_time_sec": pytest.approx(0.3),
        "total_timed_inference_sec": pytest.approx(0.6),
        "throughput_counted_images": 4,
        "throughput_images_per_sec": pytest.approx(4 / 0.6),
        "top1_index": 285,
        "top1_label": "Egyptian cat",
        "top5": [
            {
                "index": 285,
                "label": "Egyptian cat",
                "score": 0.9,
            }
        ],
    }


def test_build_manifest_image_metrics_contains_batch_position_not_timing() -> None:
    module = load_vit_example_module()

    metrics = module.build_manifest_image_metrics(
        image_path=Path("private/image_a.jpg"),
        input_shape=(3, 224, 224),
        batch_index=1,
        position_in_batch=2,
        prediction={
            "top1_index": 1,
            "top1_label": "class one",
            "top5": [{"index": 1, "label": "class one", "score": 0.8}],
        },
    )

    assert metrics == {
        "image_path": "private/image_a.jpg",
        "input_shape": [3, 224, 224],
        "batch_index": 1,
        "position_in_batch": 2,
        "top1_index": 1,
        "top1_label": "class one",
        "top5": [{"index": 1, "label": "class one", "score": 0.8}],
    }
    assert "mean_step_time_sec" not in metrics


def test_build_run_metrics_single_image_keeps_result_fields() -> None:
    module = load_vit_example_module()

    image_metrics = module.build_image_metrics(
        image_path=SAMPLE_IMAGE_PATH,
        input_shape=(2, 3, 224, 224),
        batch_size=2,
        benchmark_steps=2,
        step_times=[0.2, 0.4],
        prediction={
            "top1_index": 285,
            "top1_label": "Egyptian cat",
            "top5": [
                {
                    "index": 285,
                    "label": "Egyptian cat",
                    "score": 0.9,
                }
            ],
        },
    )
    image_metrics["predicted_index"] = 285
    image_metrics["predicted_label"] = "Egyptian cat"

    metrics = module.build_run_metrics(
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
        batch_size=2,
        warmup_steps=1,
        benchmark_steps=2,
        image_results=[image_metrics],
        manifest_path=None,
    )

    assert metrics == {
        "mode": "single_image",
        "processing_mode": "repeated_single_image",
        "num_images": 1,
        "num_batches": 1,
        "timed_batch_runs": 2,
        "num_padded_images": 0,
        "last_batch_policy": "none",
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
        "image_path": str(SAMPLE_IMAGE_PATH),
        "input_shape": [2, 3, 224, 224],
        "batch_size": 2,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "mean_step_time_sec": pytest.approx(0.3),
        "total_timed_inference_sec": pytest.approx(0.6),
        "throughput_counted_images": 4,
        "throughput_images_per_sec": pytest.approx(4 / 0.6),
        "top1_index": 285,
        "top1_label": "Egyptian cat",
        "top5": [
            {
                "index": 285,
                "label": "Egyptian cat",
                "score": 0.9,
            }
        ],
        "predicted_index": 285,
        "predicted_label": "Egyptian cat",
    }


def test_build_run_metrics_manifest_uses_aggregate_fields() -> None:
    module = load_vit_example_module()

    image_results = [
        module.build_manifest_image_metrics(
            image_path=Path("private/image_a.jpg"),
            input_shape=(3, 224, 224),
            batch_index=0,
            position_in_batch=0,
            prediction={
                "top1_index": 1,
                "top1_label": "class one",
                "top5": [{"index": 1, "label": "class one", "score": 0.8}],
            },
        ),
        module.build_manifest_image_metrics(
            image_path=Path("private/image_b.jpg"),
            input_shape=(3, 224, 224),
            batch_index=0,
            position_in_batch=1,
            prediction={
                "top1_index": 2,
                "top1_label": "class two",
                "top5": [{"index": 2, "label": "class two", "score": 0.7}],
            },
        ),
    ]

    metrics = module.build_run_metrics(
        model_name="google/vit-base-patch16-224",
        selected_jax_platform="cpu",
        backend="cpu",
        devices=[],
        batch_size=1,
        warmup_steps=1,
        benchmark_steps=2,
        image_results=image_results,
        manifest_path=PRIVATE_MANIFEST_PATH,
        input_shape=(2, 3, 224, 224),
        processing_mode="batched_manifest",
        num_batches=1,
        num_padded_images=0,
        last_batch_policy="pad_with_last_image",
        step_times=[0.1, 0.2],
    )

    assert metrics["mode"] == "image_manifest"
    assert metrics["manifest_path"] == str(PRIVATE_MANIFEST_PATH)
    assert metrics["manifest_kind"] == "local_private"
    assert metrics["input_shape"] == [2, 3, 224, 224]
    assert metrics["processing_mode"] == "batched_manifest"
    assert metrics["num_images"] == 2
    assert metrics["num_batches"] == 1
    assert metrics["timed_batch_runs"] == 2
    assert metrics["num_padded_images"] == 0
    assert metrics["last_batch_policy"] == "pad_with_last_image"
    assert metrics["mean_step_time_sec"] == pytest.approx(0.15)
    assert metrics["total_timed_inference_sec"] == pytest.approx(0.3)
    assert metrics["throughput_counted_images"] == 4
    assert metrics["throughput_images_per_sec"] == pytest.approx(4 / 0.3)
    assert metrics["image_results"] == image_results
    assert "image_path" not in metrics
    assert "predicted_label" not in metrics
    assert "top1_label" not in metrics


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


def test_collect_git_metadata_records_privacy_safe_fields(monkeypatch) -> None:
    module = load_vit_example_module()
    calls = []
    outputs = {
        ("git", "rev-parse", "HEAD"): "0123456789abcdef0123456789abcdef01234567\n",
        ("git", "symbolic-ref", "--quiet", "--short", "HEAD"): "feat/demo\n",
        ("git", "status", "--short"): "",
    }

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == module.GIT_METADATA_TIMEOUT_SEC
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return module.subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[tuple(command)],
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metadata = module.collect_git_metadata(cwd=Path("/repo"))

    assert metadata == {
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_branch": "feat/demo",
        "git_dirty": False,
    }
    assert [call[0] for call in calls] == list(outputs)


def test_collect_git_metadata_records_dirty_state(monkeypatch) -> None:
    module = load_vit_example_module()
    observed_commit = "0123456789abcdef0123456789abcdef01234567"

    def fake_run(command, **kwargs):
        outputs = {
            ("git", "rev-parse", "HEAD"): f"{observed_commit}\n",
            ("git", "symbolic-ref", "--quiet", "--short", "HEAD"): "feat/demo\n",
            ("git", "status", "--short"): " M docs/demo2_pretrained_vit.md\n",
        }
        return module.subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[tuple(command)],
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metadata = module.collect_git_metadata(cwd=Path("/repo"))

    assert metadata["git_commit"] == observed_commit
    assert metadata["git_branch"] == "feat/demo"
    assert metadata["git_dirty"] is True


def test_collect_git_metadata_returns_null_fields_when_git_unavailable(
    monkeypatch,
) -> None:
    module = load_vit_example_module()

    def fake_run(command, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metadata = module.collect_git_metadata(cwd=Path("/not-a-repo"))

    assert metadata == {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_add_cli_run_metadata_records_command_and_output_path() -> None:
    module = load_vit_example_module()

    metrics = module.add_cli_run_metadata(
        {"backend": "cpu"},
        argv=[
            "examples/pretrained_vit_inference.py",
            "--image",
            "examples/assets/chihuahua_pet_licorice.jpg",
        ],
        output_path=Path("runs/vit-inference/demo2_cpu_b1.json"),
        git_metadata={
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_branch": "feat/demo",
            "git_dirty": True,
        },
    )

    assert metrics["backend"] == "cpu"
    assert metrics["command_used"] == (
        "python examples/pretrained_vit_inference.py "
        "--image examples/assets/chihuahua_pet_licorice.jpg"
    )
    assert metrics["output_path"] == "runs/vit-inference/demo2_cpu_b1.json"
    assert metrics["git_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert metrics["git_branch"] == "feat/demo"
    assert metrics["git_dirty"] is True
