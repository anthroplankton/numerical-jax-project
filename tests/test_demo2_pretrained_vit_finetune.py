from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SCRIPT_PATH = EXAMPLES_DIR / "demo2_pretrained_vit_finetune.py"
TRAIN_MANIFEST = Path("data/local/imagenette2-320/train/manifest_train_64.txt")
EVAL_MANIFEST = Path("data/local/imagenette2-320/val/manifest_val_64.txt")


def load_finetune_module():
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(
        "demo2_pretrained_vit_finetune", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_module_import_does_not_import_jax_before_platform_selection() -> None:
    probe = """
import importlib.util
import json
import sys
from pathlib import Path

examples_dir = Path("examples").resolve()
sys.path.insert(0, str(examples_dir))
script_path = examples_dir / "demo2_pretrained_vit_finetune.py"
spec = importlib.util.spec_from_file_location("finetune_import_probe", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
loaded = {
    "jax": "jax" in sys.modules,
    "jax.sharding": "jax.sharding" in sys.modules,
}
print(json.dumps(loaded, sort_keys=True))
raise SystemExit(1 if any(loaded.values()) else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"jax": False, "jax.sharding": False}


def test_parse_args_uses_small_cpu_safe_defaults() -> None:
    module = load_finetune_module()

    args = module.parse_args(
        [
            "--train-manifest",
            str(TRAIN_MANIFEST),
            "--eval-manifest",
            str(EVAL_MANIFEST),
        ]
    )

    assert args.model == "google/vit-base-patch16-224"
    assert args.train_manifest == TRAIN_MANIFEST
    assert args.eval_manifest == EVAL_MANIFEST
    assert args.batch_size == 8
    assert args.batch_sharding == "none"
    assert args.mesh_axis_name == "data"
    assert args.require_multiple_devices is False
    assert args.min_shard_devices == 2
    assert args.learning_rate == pytest.approx(1e-3)
    assert args.max_steps == 20
    assert args.min_train_seconds == 0.0
    assert args.checkpoint_every_steps == 10
    assert args.checkpoint_every_seconds == 30.0
    assert args.eval_every_steps == 0
    assert args.checkpoint_dir == Path(
        "runs/vit-finetune/demo2_vit_head_finetune/checkpoints"
    )
    assert args.output_dir == Path("runs/vit-finetune/demo2_vit_head_finetune")
    assert args.jax_platform == "cpu"
    assert args.reinit_head is False
    assert args.seed == 0
    assert args.resume is False
    assert args.save_predictions is False


def test_parse_args_accepts_tpu_resume_and_time_controlled_settings() -> None:
    module = load_finetune_module()

    args = module.parse_args(
        [
            "--train-manifest",
            str(TRAIN_MANIFEST),
            "--eval-manifest",
            str(EVAL_MANIFEST),
            "--jax-platform",
            "tpu",
            "--batch-sharding",
            "data",
            "--mesh-axis-name",
            "batch",
            "--require-multiple-devices",
            "--min-shard-devices",
            "4",
            "--max-steps",
            "100000",
            "--min-train-seconds",
            "120",
            "--checkpoint-every-steps",
            "20",
            "--checkpoint-every-seconds",
            "30",
            "--eval-every-steps",
            "25",
            "--checkpoint-dir",
            "runs/vit-finetune/cloud/checkpoints",
            "--output-dir",
            "runs/vit-finetune/cloud",
            "--reinit-head",
            "--seed",
            "123",
            "--resume",
            "--save-predictions",
        ]
    )

    assert args.jax_platform == "tpu"
    assert args.batch_sharding == "data"
    assert args.mesh_axis_name == "batch"
    assert args.require_multiple_devices is True
    assert args.min_shard_devices == 4
    assert args.max_steps == 100000
    assert args.min_train_seconds == pytest.approx(120.0)
    assert args.checkpoint_every_steps == 20
    assert args.checkpoint_every_seconds == pytest.approx(30.0)
    assert args.eval_every_steps == 25
    assert args.checkpoint_dir == Path("runs/vit-finetune/cloud/checkpoints")
    assert args.output_dir == Path("runs/vit-finetune/cloud")
    assert args.reinit_head is True
    assert args.seed == 123
    assert args.resume is True
    assert args.save_predictions is True


def test_parse_args_rejects_invalid_sharding_settings() -> None:
    module = load_finetune_module()

    base_args = [
        "--train-manifest",
        str(TRAIN_MANIFEST),
        "--eval-manifest",
        str(EVAL_MANIFEST),
    ]
    with pytest.raises(SystemExit):
        module.parse_args([*base_args, "--batch-sharding", "model"])

    with pytest.raises(SystemExit):
        module.parse_args([*base_args, "--min-shard-devices", "0"])


def test_resolve_run_paths_makes_output_and_checkpoint_dirs_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_finetune_module()
    monkeypatch.chdir(PROJECT_ROOT)
    args = module.parse_args(
        [
            "--train-manifest",
            str(TRAIN_MANIFEST),
            "--eval-manifest",
            str(EVAL_MANIFEST),
            "--checkpoint-dir",
            "runs/vit-finetune/cloud/checkpoints",
            "--output-dir",
            "runs/vit-finetune/cloud",
        ]
    )

    resolved = module.resolve_run_paths(args)

    assert resolved.checkpoint_dir.is_absolute()
    assert resolved.output_dir.is_absolute()
    assert (
        resolved.checkpoint_dir
        == (PROJECT_ROOT / "runs/vit-finetune/cloud/checkpoints").resolve()
    )
    assert resolved.output_dir == (PROJECT_ROOT / "runs/vit-finetune/cloud").resolve()


def test_read_labeled_manifest_derives_imagenette_labels(tmp_path) -> None:
    module = load_finetune_module()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "# path-only Imagenette manifest",
                "n01440764/image_a.JPEG",
                "n03445777/image_b.JPEG",
                "",
            ]
        )
    )

    entries = module.read_labeled_manifest(manifest)

    assert entries == [
        module.LabeledImage(
            image_path=tmp_path / "n01440764" / "image_a.JPEG",
            label_id="n01440764",
            label_index=0,
        ),
        module.LabeledImage(
            image_path=tmp_path / "n03445777" / "image_b.JPEG",
            label_id="n03445777",
            label_index=574,
        ),
    ]


def test_read_labeled_manifest_rejects_unknown_parent_label(tmp_path) -> None:
    module = load_finetune_module()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("not_imagenette/image.JPEG\n")

    with pytest.raises(ValueError, match="unsupported Imagenette label directory"):
        module.read_labeled_manifest(manifest)


def test_count_labels_returns_sorted_label_counts(tmp_path) -> None:
    module = load_finetune_module()
    entries = [
        module.LabeledImage(tmp_path / "n03445777" / "a.JPEG", "n03445777", 574),
        module.LabeledImage(tmp_path / "n01440764" / "b.JPEG", "n01440764", 0),
        module.LabeledImage(tmp_path / "n03445777" / "c.JPEG", "n03445777", 574),
    ]

    assert module.count_labels(entries) == {
        "n01440764": 1,
        "n03445777": 2,
    }


def test_pad_label_batch_repeats_last_real_label() -> None:
    module = load_finetune_module()

    padded = module.pad_label_batch(
        np.asarray([0, 217]),
        padding_count=2,
        jnp_module=np,
    )

    np.testing.assert_array_equal(padded, np.asarray([0, 217, 217, 217]))


def test_build_batch_mask_marks_padding() -> None:
    module = load_finetune_module()

    mask = module.build_batch_mask(
        real_size=3,
        padding_count=1,
        jnp_module=np,
    )

    np.testing.assert_array_equal(mask, np.asarray([1.0, 1.0, 1.0, 0.0]))


def test_shard_training_batches_routes_arrays_through_resolved_sharding() -> None:
    module = load_finetune_module()

    class FakeResolvedSharding:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def shard_image_batch(self, value, *, jax_module):
            self.calls.append(f"image:{jax_module}")
            return ("image", value)

        def shard_label_batch(self, value, *, jax_module):
            self.calls.append(f"label:{jax_module}")
            return ("label", value)

        def shard_mask_batch(self, value, *, jax_module):
            self.calls.append(f"mask:{jax_module}")
            return ("mask", value)

    resolved_sharding = FakeResolvedSharding()
    batch = module.TrainingBatch(
        pixel_values=np.asarray([[1.0]]),
        labels=np.asarray([1]),
        mask=np.asarray([1.0]),
        start_index=3,
        real_size=1,
        padding_count=0,
    )

    sharded = module.shard_training_batches(
        [batch],
        resolved_sharding=resolved_sharding,
        jax_module="jax",
    )

    assert resolved_sharding.calls == ["image:jax", "label:jax", "mask:jax"]
    assert len(sharded) == 1
    assert sharded[0].pixel_values[0] == "image"
    assert sharded[0].labels[0] == "label"
    assert sharded[0].mask[0] == "mask"
    np.testing.assert_array_equal(sharded[0].pixel_values[1], batch.pixel_values)
    np.testing.assert_array_equal(sharded[0].labels[1], batch.labels)
    np.testing.assert_array_equal(sharded[0].mask[1], batch.mask)
    assert sharded[0].start_index == 3
    assert sharded[0].real_size == 1
    assert sharded[0].padding_count == 0


def test_training_loop_continuation_uses_time_control_as_upper_bound_shape() -> None:
    module = load_finetune_module()

    assert module.should_continue_training(
        step=5,
        max_steps=100000,
        elapsed_sec=30,
        min_train_seconds=120,
    )
    assert not module.should_continue_training(
        step=5,
        max_steps=100000,
        elapsed_sec=121,
        min_train_seconds=120,
    )
    assert not module.should_continue_training(
        step=100000,
        max_steps=100000,
        elapsed_sec=30,
        min_train_seconds=120,
    )


def test_checkpoint_schedule_supports_step_time_and_sigterm() -> None:
    module = load_finetune_module()

    assert module.should_save_checkpoint(
        step=20,
        now=35,
        last_checkpoint_time=10,
        checkpoint_every_steps=20,
        checkpoint_every_seconds=30,
    )
    assert module.should_save_checkpoint(
        step=3,
        now=45,
        last_checkpoint_time=10,
        checkpoint_every_steps=20,
        checkpoint_every_seconds=30,
    )
    assert module.should_save_checkpoint(
        step=3,
        now=11,
        last_checkpoint_time=10,
        checkpoint_every_steps=20,
        checkpoint_every_seconds=30,
        stop_requested=True,
    )
    assert not module.should_save_checkpoint(
        step=0,
        now=100,
        last_checkpoint_time=0,
        checkpoint_every_steps=1,
        checkpoint_every_seconds=1,
        stop_requested=True,
    )


def test_periodic_eval_schedule_is_step_based() -> None:
    module = load_finetune_module()

    assert module.should_run_periodic_eval(
        step=25,
        start_step=0,
        eval_every_steps=25,
    )
    assert not module.should_run_periodic_eval(
        step=0,
        start_step=0,
        eval_every_steps=25,
    )
    assert not module.should_run_periodic_eval(
        step=30,
        start_step=0,
        eval_every_steps=25,
    )
    assert not module.should_run_periodic_eval(
        step=25,
        start_step=0,
        eval_every_steps=0,
    )


def test_eval_metrics_csv_writes_pandas_friendly_columns(tmp_path) -> None:
    module = load_finetune_module()
    metrics_path = tmp_path / "eval_metrics.csv"

    module.write_eval_metrics_header(metrics_path)
    module.append_eval_metrics_row(
        metrics_path,
        step=5,
        metrics={"loss": 1.25, "accuracy": 0.5},
    )

    assert metrics_path.read_text().splitlines() == [
        "step,eval_loss,eval_accuracy",
        "5,1.25,0.5",
    ]


def test_reinitialize_classifier_head_preserves_shape_and_zeroes_bias() -> None:
    import jax
    import jax.numpy as jnp

    module = load_finetune_module()
    head_params = {
        "kernel": jnp.ones((2, 3)),
        "bias": jnp.ones((3,)),
    }

    reinitialized = module.reinitialize_classifier_head(
        head_params,
        seed=0,
        jax_module=jax,
        jnp_module=jnp,
    )
    reinitialized_again = module.reinitialize_classifier_head(
        head_params,
        seed=0,
        jax_module=jax,
        jnp_module=jnp,
    )

    assert reinitialized["kernel"].shape == head_params["kernel"].shape
    assert reinitialized["bias"].shape == head_params["bias"].shape
    assert not np.allclose(np.asarray(reinitialized["kernel"]), np.ones((2, 3)))
    np.testing.assert_allclose(np.asarray(reinitialized["bias"]), np.zeros((3,)))
    np.testing.assert_allclose(
        np.asarray(reinitialized["kernel"]),
        np.asarray(reinitialized_again["kernel"]),
    )


def test_reset_signal_state_clears_stale_in_process_stop_request() -> None:
    module = load_finetune_module()
    module.STOP_REQUESTED = True
    module.STOP_SIGNAL_NAME = "SIGTERM"

    module.reset_signal_state()

    assert module.STOP_REQUESTED is False
    assert module.STOP_SIGNAL_NAME is None


def test_checkpoint_payload_excludes_frozen_backbone() -> None:
    module = load_finetune_module()
    item = module.build_checkpoint_item(
        head_params={"kernel": np.zeros((2, 3)), "bias": np.zeros((3,))},
        optimizer_state={"count": np.asarray(0)},
        step=4,
        metadata={"mode": module.MODE, "git_commit": "abc"},
    )

    assert module.checkpoint_contains_allowed_keys(item)
    assert "vit" not in item
    assert "backbone" not in item
    assert int(item["step"]) == 4
    metadata = module.decode_checkpoint_metadata(item["metadata_json"])
    assert metadata == {"git_commit": "abc", "mode": module.MODE}


def test_validate_restored_checkpoint_metadata_rejects_identity_mismatch() -> None:
    module = load_finetune_module()
    expected = {
        "mode": module.MODE,
        "model_name": "google/vit-base-patch16-224",
        "trainable_scope": module.TRAINABLE_SCOPE,
        "frozen_scope": module.FROZEN_SCOPE,
    }

    module.validate_restored_checkpoint_metadata(dict(expected), expected)

    restored = {
        **expected,
        "model_name": "different/model",
    }
    with pytest.raises(ValueError, match="checkpoint metadata mismatch"):
        module.validate_restored_checkpoint_metadata(restored, expected)


def test_checkpoint_identity_validation_ignores_sharding_metadata() -> None:
    module = load_finetune_module()
    expected = {
        "mode": module.MODE,
        "model_name": "google/vit-base-patch16-224",
        "trainable_scope": module.TRAINABLE_SCOPE,
        "frozen_scope": module.FROZEN_SCOPE,
        "sharding": {"mode": "data"},
    }
    restored = {
        **expected,
        "sharding": {"mode": "none"},
    }

    module.validate_restored_checkpoint_metadata(restored, expected)


def test_build_summary_has_expected_schema() -> None:
    module = load_finetune_module()

    summary = module.build_summary(
        model_name="google/vit-base-patch16-224",
        selected_jax_platform="cpu",
        backend="cpu",
        devices=[{"platform": "cpu", "device_kind": "cpu", "id": 0, "repr": "cpu"}],
        train_manifest=TRAIN_MANIFEST,
        eval_manifest=EVAL_MANIFEST,
        train_examples=64,
        eval_examples=64,
        train_label_counts={"n01440764": 32, "n03445777": 32},
        eval_label_counts={"n01440764": 16, "n03445777": 48},
        batch_size=8,
        learning_rate=1e-3,
        eval_every_steps=25,
        reinit_head=True,
        seed=123,
        start_step=10,
        final_step=20,
        resumed_from_checkpoint=True,
        checkpoint_dir=Path("runs/vit-finetune/checkpoints"),
        latest_checkpoint_step=20,
        initial_loss=2.0,
        final_loss=1.5,
        step_times=[0.1, 0.2],
        total_train_examples=16,
        total_runtime_sec=3.0,
        interrupted=False,
        stop_signal_name=None,
        git_metadata={
            "git_commit": "0123",
            "git_branch": "main",
            "git_dirty": False,
        },
        sharding_metadata={
            "enabled": True,
            "mode": "data",
            "mesh_axis_name": "data",
            "label_partition_spec": "PartitionSpec('data')",
            "mask_partition_spec": "PartitionSpec('data')",
        },
    )

    assert summary["mode"] == "demo2_vit_head_finetune"
    assert summary["trainable_scope"] == "classifier_head_only"
    assert summary["frozen_scope"] == "vit_backbone"
    assert summary["train_label_counts"] == {"n01440764": 32, "n03445777": 32}
    assert summary["eval_label_counts"] == {"n01440764": 16, "n03445777": 48}
    assert summary["num_train_classes"] == 2
    assert summary["num_eval_classes"] == 2
    assert summary["sharding"] == {
        "enabled": True,
        "mode": "data",
        "mesh_axis_name": "data",
        "label_partition_spec": "PartitionSpec('data')",
        "mask_partition_spec": "PartitionSpec('data')",
    }
    assert summary["eval_every_steps"] == 25
    assert summary["reinit_head"] is True
    assert summary["seed"] == 123
    assert summary["resumed_from_checkpoint"] is True
    assert summary["initial_loss"] == pytest.approx(2.0)
    assert summary["final_loss"] == pytest.approx(1.5)
    assert summary["mean_step_time_sec"] == pytest.approx(0.15)
    assert summary["examples_per_second"] == pytest.approx(16 / 0.3)
    assert (
        "excludes checkpoint write time"
        in summary["timing_scope"]["mean_step_time_sec"]
    )
    assert "full run wall time" in summary["timing_scope"]["total_runtime_sec"]
    assert summary["git_commit"] == "0123"
