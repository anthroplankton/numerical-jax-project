# Numerical JAX Project

Course project repository for Numerical Computation with JAX. The initial focus
is validating the local JAX runtime and keeping a small foundation for later
numerical experiments.

## Local Sanity Check

Run the JAX device summary script first:

```bash
bash scripts/check_jax_device.sh
```

You can also run the package CLI directly:

```bash
uv run python -m jax_tpu_project.cli devices
```

The command prints the active JAX backend and visible devices as JSON. It should
work on CPU-only machines and GPU-enabled machines.
