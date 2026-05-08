"""Command-line interface for the Numerical JAX project."""

from __future__ import annotations

import json

import typer

from jax_tpu_project.runtime import get_device_summary

app = typer.Typer(help="Utilities for the Numerical Computation with JAX project.")


@app.callback()
def root() -> None:
    """Numerical Computation with JAX course project utilities."""


@app.command()
def devices() -> None:
    """Print the local JAX runtime device summary as JSON."""
    typer.echo(json.dumps(get_device_summary(), indent=2, sort_keys=True))


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
