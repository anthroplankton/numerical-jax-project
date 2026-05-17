from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_image_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("build_image_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_find_image_paths_is_sorted_limited_and_extension_filtered(tmp_path) -> None:
    module = load_manifest_module()
    touch(tmp_path / "b.PNG")
    touch(tmp_path / "a.jpg")
    touch(tmp_path / "nested" / "c.jpeg")
    touch(tmp_path / "notes.txt")

    image_paths = module.find_image_paths(
        tmp_path,
        extensions=(".jpg", ".jpeg", ".png"),
        limit=2,
    )

    assert image_paths == [
        tmp_path / "a.jpg",
        tmp_path / "b.PNG",
    ]


def test_build_manifest_text_uses_paths_relative_to_manifest_dir(tmp_path) -> None:
    module = load_manifest_module()
    image_paths = [
        tmp_path / "a.jpg",
        tmp_path / "nested" / "b.png",
    ]
    manifest_path = tmp_path / "manifest.txt"

    manifest_text = module.build_manifest_text(image_paths, manifest_path)

    assert str(tmp_path) not in manifest_text
    assert manifest_text.splitlines() == [
        "# Demo 2 image manifest generated from existing local files.",
        "a.jpg",
        "nested/b.png",
    ]


def test_write_manifest_creates_parent_directory(tmp_path) -> None:
    module = load_manifest_module()
    image_path = tmp_path / "images" / "sample.jpg"
    manifest_path = tmp_path / "manifests" / "manifest.txt"

    module.write_manifest([image_path], manifest_path)

    assert manifest_path.read_text().splitlines()[-1] == "../images/sample.jpg"


def test_find_image_paths_rejects_empty_directory(tmp_path) -> None:
    module = load_manifest_module()

    with pytest.raises(ValueError, match="no images"):
        module.find_image_paths(tmp_path, extensions=(".jpg",))
