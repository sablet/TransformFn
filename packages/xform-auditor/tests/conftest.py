from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterator

import pytest


def _ensure_package_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    packages_dir = repo_root / "packages"
    for package in (
        packages_dir / "xform-core",
        packages_dir / "proj-dtypes",
        packages_dir / "xform-auditor",
    ):
        path_str = str(package)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_package_paths()


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    from xform_core import check_registry, example_registry

    example_registry.clear()
    check_registry.clear()
    try:
        yield
    finally:
        example_registry.clear()
        check_registry.clear()


@pytest.fixture()
def module_dir(tmp_path: Path) -> Iterator[Path]:
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield tmp_path
    finally:
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()
