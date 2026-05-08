"""Run the raw-JAX CNN benchmark on MNIST-shaped data.

Usage:
    uv run python examples/cnn_mnist_benchmark.py --dataset synthetic --steps 5

The default synthetic dataset needs no network access. The CNN is implemented by
hand with JAX primitives so the same training step can later run on TPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep the documented smoke command CPU-safe unless the caller chooses a backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax_tpu_project.cnn_mnist import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STEPS,
    DEFAULT_WARMUP_STEPS,
    SUPPORTED_DATASETS,
    format_benchmark_summary,
    run_cnn_mnist_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="synthetic",
        help="Dataset to use. Real datasets are reserved for local-file support.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for future local MNIST/Fashion-MNIST files.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of timed training steps. Default: {DEFAULT_STEPS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for training. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"SGD learning rate. Default: {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for parameters and synthetic data. Default: 0.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Untimed warmup steps before benchmarking. Default: {DEFAULT_WARMUP_STEPS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON metrics. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--platform-label",
        default="local",
        help="Free-form label for the run environment. Default: local.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print per-step metrics every N timed steps. Default: 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_cnn_mnist_benchmark(
        dataset=args.dataset,
        data_dir=args.data_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        platform_label=args.platform_label,
        log_every=args.log_every,
    )
    print(format_benchmark_summary(metrics))


if __name__ == "__main__":
    main()
