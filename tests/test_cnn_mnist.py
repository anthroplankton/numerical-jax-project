from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from jax_tpu_project.cnn_mnist import (
    IMAGE_SHAPE,
    NUM_CLASSES,
    forward,
    init_cnn_params,
    make_synthetic_mnist_data,
    run_cnn_mnist_benchmark,
    training_step,
)


def test_parameter_initialization_returns_expected_shapes() -> None:
    params = init_cnn_params(seed=0)

    assert params["conv1"]["kernel"].shape == (3, 3, 1, 8)
    assert params["conv1"]["bias"].shape == (8,)
    assert params["conv2"]["kernel"].shape == (3, 3, 8, 16)
    assert params["conv2"]["bias"].shape == (16,)
    assert params["dense1"]["kernel"].shape == (7 * 7 * 16, 32)
    assert params["dense1"]["bias"].shape == (32,)
    assert params["logits"]["kernel"].shape == (32, NUM_CLASSES)
    assert params["logits"]["bias"].shape == (NUM_CLASSES,)


def test_forward_pass_returns_logits_for_each_example() -> None:
    params = init_cnn_params(seed=0)
    images, _ = make_synthetic_mnist_data(num_examples=4, seed=1)

    logits = forward(params, images)

    assert logits.shape == (4, NUM_CLASSES)


def test_training_steps_run_without_error() -> None:
    params = init_cnn_params(seed=0)
    images, labels = make_synthetic_mnist_data(num_examples=8, seed=2)

    params, first_loss, first_accuracy = training_step(
        params, images[:4], labels[:4], 0.05
    )
    params, second_loss, second_accuracy = training_step(
        params, images[4:], labels[4:], 0.05
    )
    first_loss.block_until_ready()
    second_loss.block_until_ready()

    assert jnp.isfinite(first_loss)
    assert jnp.isfinite(second_loss)
    assert 0.0 <= float(first_accuracy) <= 1.0
    assert 0.0 <= float(second_accuracy) <= 1.0


def test_benchmark_writes_metrics_json(tmp_path) -> None:
    metrics = run_cnn_mnist_benchmark(
        dataset="synthetic",
        steps=1,
        batch_size=4,
        seed=0,
        warmup_steps=0,
        output_dir=tmp_path,
        platform_label="test",
        log_every=1,
    )
    metrics_path = tmp_path / "cnn_mnist_metrics.json"

    assert metrics_path.exists()
    written_metrics = json.loads(metrics_path.read_text())
    assert written_metrics["output_artifact_path"] == str(metrics_path)
    assert written_metrics["dataset_name"] == "synthetic"
    assert written_metrics["platform_label"] == "test"
    assert written_metrics["batch_size"] == 4
    assert written_metrics["steps"] == 1
    assert written_metrics["warmup_steps"] == 0
    assert metrics["output_artifact_path"] == str(metrics_path)


def test_fixed_seed_behavior_is_deterministic_enough(tmp_path) -> None:
    first_images, first_labels = make_synthetic_mnist_data(num_examples=6, seed=7)
    second_images, second_labels = make_synthetic_mnist_data(num_examples=6, seed=7)

    assert first_images.shape == (6, *IMAGE_SHAPE)
    assert jnp.array_equal(first_images, second_images)
    assert jnp.array_equal(first_labels, second_labels)

    first_metrics = run_cnn_mnist_benchmark(
        dataset="synthetic",
        steps=1,
        batch_size=4,
        seed=7,
        warmup_steps=0,
        output_dir=tmp_path / "first",
        platform_label="test",
        log_every=1,
    )
    second_metrics = run_cnn_mnist_benchmark(
        dataset="synthetic",
        steps=1,
        batch_size=4,
        seed=7,
        warmup_steps=0,
        output_dir=tmp_path / "second",
        platform_label="test",
        log_every=1,
    )

    assert first_metrics["initial_loss"] == pytest.approx(
        second_metrics["initial_loss"]
    )
    assert first_metrics["final_loss"] == pytest.approx(second_metrics["final_loss"])
    assert first_metrics["final_accuracy"] == pytest.approx(
        second_metrics["final_accuracy"]
    )
