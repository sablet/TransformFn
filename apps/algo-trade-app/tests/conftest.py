from __future__ import annotations

import pytest

from xform_core import clear_registries


@pytest.fixture(autouse=True)
def _reset_type_registries() -> None:
    clear_registries()
    yield
    clear_registries()
