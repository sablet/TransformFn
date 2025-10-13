from __future__ import annotations

import pytest

from xform_core import clear_registries
from proj_dtypes.registry_setup import register_defaults


@pytest.fixture(autouse=True)
def _reset_type_registry() -> None:
    clear_registries()
    register_defaults.cache_clear()
    yield
    clear_registries()
    register_defaults.cache_clear()
    register_defaults()
