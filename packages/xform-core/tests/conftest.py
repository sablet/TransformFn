from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest


def _ensure_package_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_dir = repo_root / "packages" / "xform-core"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


_ensure_package_on_path()


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    from xform_core import check_registry, clear_registries, example_registry

    example_registry.clear()
    check_registry.clear()
    clear_registries()
    yield
    example_registry.clear()
    check_registry.clear()
    clear_registries()
