"""Course project package for Numerical Computation with JAX."""

from jax_tpu_project.cnn_mnist import run_cnn_mnist_benchmark
from jax_tpu_project.runtime import get_device_summary

__all__ = ["get_device_summary", "run_cnn_mnist_benchmark"]
