"""Build a deterministic local image manifest for Demo 2 inputs.

This helper scans an existing local directory and writes a plain text manifest
with one image path per line. It does not download datasets or inspect image
contents, so it is safe for local-only Imagenette or private-demo preparation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_HEADER = "# Demo 2 image manifest generated from existing local files."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the manifest builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Existing local image directory to scan recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path to write. Default: <image_dir>/manifest.txt.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Optional maximum number of sorted image paths to include.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=_positive_int,
        help=(
            "Optional class-balanced mode: select up to N sorted images from "
            "each immediate parent directory."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help=(
            "Image file extensions to include. "
            f"Default: {' '.join(DEFAULT_EXTENSIONS)}."
        ),
    )
    return parser.parse_args(argv)


def normalize_extensions(raw_extensions: Sequence[str]) -> tuple[str, ...]:
    """Normalize extension filters to lowercase dot-prefixed suffixes."""
    extensions = tuple(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in raw_extensions
    )
    if not extensions:
        msg = "at least one image extension is required"
        raise ValueError(msg)
    return extensions


def find_image_paths(
    image_dir: Path,
    *,
    extensions: Sequence[str],
    limit: int | None = None,
    per_class_limit: int | None = None,
) -> list[Path]:
    """Return sorted local image paths without opening the files."""
    if limit is not None and per_class_limit is not None:
        msg = "use either limit or per_class_limit, not both"
        raise ValueError(msg)
    if not image_dir.is_dir():
        msg = f"image directory does not exist: {image_dir}"
        raise NotADirectoryError(msg)

    normalized_extensions = normalize_extensions(extensions)
    image_paths = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if per_class_limit is not None:
        image_paths = select_per_parent_class(
            image_paths,
            per_class_limit=per_class_limit,
        )
    elif limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        extensions_text = ", ".join(normalized_extensions)
        msg = f"no images with extensions {extensions_text} under {image_dir}"
        raise ValueError(msg)
    return image_paths


def select_per_parent_class(
    image_paths: Sequence[Path],
    *,
    per_class_limit: int,
) -> list[Path]:
    """Select up to N sorted images from each immediate parent directory."""
    grouped_paths: dict[Path, list[Path]] = {}
    for image_path in image_paths:
        grouped_paths.setdefault(image_path.parent, []).append(image_path)

    selected: list[Path] = []
    for class_dir in sorted(grouped_paths, key=lambda path: path.as_posix().lower()):
        selected.extend(grouped_paths[class_dir][:per_class_limit])
    return selected


def build_manifest_text(image_paths: Sequence[Path], manifest_path: Path) -> str:
    """Build manifest text using paths relative to the manifest directory."""
    manifest_dir = manifest_path.parent
    lines = [DEFAULT_HEADER]
    for image_path in image_paths:
        entry = Path(os.path.relpath(image_path, manifest_dir))
        lines.append(entry.as_posix())
    return "\n".join(lines) + "\n"


def write_manifest(image_paths: Sequence[Path], manifest_path: Path) -> None:
    """Write a manifest file for the provided image paths."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(build_manifest_text(image_paths, manifest_path))


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        msg = "must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return value


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    output_path = args.output or args.image_dir / "manifest.txt"
    image_paths = find_image_paths(
        args.image_dir,
        extensions=args.extensions,
        limit=args.limit,
        per_class_limit=args.per_class_limit,
    )
    write_manifest(image_paths, output_path)
    print(f"manifest: {output_path}")
    print(f"images: {len(image_paths)}")


if __name__ == "__main__":
    main()
