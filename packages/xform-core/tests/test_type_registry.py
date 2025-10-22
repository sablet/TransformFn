from __future__ import annotations

from dataclasses import dataclass

import pytest

from xform_core import (
    Check,
    ExampleValue,
    MissingCheckError,
    MissingExampleError,
    RegisteredType,
    clear_registries,
    ensure_checks,
    ensure_examples,
    list_check_keys,
    list_example_keys,
    register_check,
    register_example,
    resolve_checks,
    resolve_examples,
)
from xform_core.type_registry import TypeRegistryError, ensure_registered

_EXPECTED_CHECK_COUNT = 2


@pytest.fixture(autouse=True)
def _reset_type_registries() -> None:
    clear_registries()


def test_register_and_resolve_examples_preserves_order() -> None:
    key = "collections.Counter"
    register_example(key, ExampleValue({"a": 1}))
    register_example(key, ExampleValue({"b": 2}))

    resolved = resolve_examples(key)

    assert [example.value for example in resolved] == [{"a": 1}, {"b": 2}]


def test_register_examples_avoids_duplicates() -> None:
    key = "builtins.int"
    example = ExampleValue(1)

    register_example(key, example)
    register_example(key, example)

    resolved = resolve_examples(key)
    assert len(resolved) == 1


def test_ensure_examples_raises_missing_example_error() -> None:
    with pytest.raises(MissingExampleError) as excinfo:
        ensure_examples("missing.Key", param_name="payload")

    message = str(excinfo.value)
    assert "payload" in message
    assert "missing.Key" in message


def test_register_and_resolve_checks() -> None:
    key = "algo_trade_dtypes.types.FeatureMap"
    check = Check("algo_trade_dtypes.checks.verify")

    register_check(key, check)
    register_check(key, Check("algo_trade_dtypes.checks.other"))

    resolved = resolve_checks(key)
    assert resolved[0] is check
    assert len(resolved) == _EXPECTED_CHECK_COUNT
    assert list_check_keys()


def test_ensure_checks_raises_missing_check_error() -> None:
    with pytest.raises(MissingCheckError) as excinfo:
        ensure_checks("algo_trade_dtypes.types.Unknown")

    assert "Unknown" in str(excinfo.value)


def test_list_example_keys_reflects_registrations() -> None:
    register_example("builtins.str", ExampleValue("hello"))

    assert "builtins.str" in list_example_keys()


def test_registered_type_allows_builtin_types() -> None:
    RegisteredType(int).with_example(1).register()
    ensure_examples("builtins.int")
    ensure_registered(int, context="payload")


def test_registered_type_allows_local_structures() -> None:
    @dataclass
    class LocalSchema:
        value: int

    RegisteredType(LocalSchema).with_example(LocalSchema(value=1)).register()
    keys = list_example_keys()
    assert any(key.endswith("LocalSchema") for key in keys)


def test_registered_type_requires_repo_defined_structured_alias() -> None:
    class ExternalFrame:
        pass

    ExternalFrame.__module__ = "external.lib"

    with pytest.warns(UserWarning):
        RegisteredType(ExternalFrame).with_example(ExternalFrame()).register()

    with pytest.raises(TypeRegistryError):
        ensure_registered(ExternalFrame, context="payload")


def test_registered_type_blocks_third_party_without_alias() -> None:
    class ForeignFrame:
        pass

    ForeignFrame.__module__ = "external.lib"

    with pytest.warns(UserWarning):
        RegisteredType(ForeignFrame).with_example(ForeignFrame()).register()

    with pytest.raises(TypeRegistryError):
        ensure_registered(ForeignFrame, context="payload")
