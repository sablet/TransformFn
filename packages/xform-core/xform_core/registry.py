"""型ベースの Example / Check レジストリ実装。"""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Dict, Iterator, List, Tuple

from .exceptions import MissingCheckError, MissingExampleError
from .metadata import Check, ExampleValue

ExampleKey = str


class _RegistryBase:
    def __init__(self) -> None:
        self._lock = RLock()

    def _validate_key(self, key: ExampleKey) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("registry key must be a non-empty string")

    def _dedupe_append(self, items: list[Any], value: Any) -> None:  # noqa: ANN401
        for existing in items:
            if existing is value:
                return
            try:
                if existing == value:
                    return
            except ValueError:
                continue
        items.append(value)


class ExampleRegistry(_RegistryBase):
    """型 FQN をキーとして ExampleValue を保持するレジストリ。"""

    def __init__(self) -> None:
        super().__init__()
        self._storage: Dict[ExampleKey, List[ExampleValue[object]]] = defaultdict(list)

    def register(self, key: ExampleKey, example: ExampleValue[object]) -> None:
        self._validate_key(key)
        if not isinstance(example, ExampleValue):
            raise TypeError("example must be an ExampleValue instance")
        with self._lock:
            self._dedupe_append(self._storage[key], example)

    def resolve(self, key: ExampleKey) -> Tuple[ExampleValue[object], ...]:
        with self._lock:
            return tuple(self._storage.get(key, ()))

    def ensure(
        self, key: ExampleKey, *, param_name: str | None = None
    ) -> Tuple[ExampleValue[object], ...]:
        examples = self.resolve(key)
        if not examples:
            available = tuple(self.iter_keys())
            raise MissingExampleError(
                key=key,
                param_name=param_name,
                available_keys=available,
            )
        return examples

    def iter_keys(self) -> Iterator[ExampleKey]:
        with self._lock:
            return iter(tuple(self._storage.keys()))

    def items(self) -> Tuple[Tuple[ExampleKey, Tuple[ExampleValue[object], ...]], ...]:
        with self._lock:
            return tuple((key, tuple(values)) for key, values in self._storage.items())

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()


class CheckRegistry(_RegistryBase):
    """型 FQN をキーとして Check を保持するレジストリ。"""

    def __init__(self) -> None:
        super().__init__()
        self._storage: Dict[ExampleKey, List[Check]] = defaultdict(list)

    def register(self, key: ExampleKey, check: Check) -> None:
        self._validate_key(key)
        if not isinstance(check, Check):
            raise TypeError("check must be a Check instance")
        with self._lock:
            self._dedupe_append(self._storage[key], check)

    def resolve(self, key: ExampleKey) -> Tuple[Check, ...]:
        with self._lock:
            return tuple(self._storage.get(key, ()))

    def ensure(self, key: ExampleKey, *, slot: str = "output") -> Tuple[Check, ...]:
        checks = self.resolve(key)
        if not checks:
            available = tuple(self.iter_keys())
            raise MissingCheckError(key=key, slot=slot, available_keys=available)
        return checks

    def iter_keys(self) -> Iterator[ExampleKey]:
        with self._lock:
            return iter(tuple(self._storage.keys()))

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()

    def items(self) -> Tuple[Tuple[ExampleKey, Tuple[Check, ...]], ...]:
        with self._lock:
            return tuple((key, tuple(values)) for key, values in self._storage.items())


example_registry = ExampleRegistry()
check_registry = CheckRegistry()


def register_example(key: ExampleKey, example: ExampleValue[object]) -> None:
    example_registry.register(key, example)


def register_check(key: ExampleKey, check: Check) -> None:
    check_registry.register(key, check)


def resolve_examples(key: ExampleKey) -> Tuple[ExampleValue[object], ...]:
    return example_registry.resolve(key)


def resolve_checks(key: ExampleKey) -> Tuple[Check, ...]:
    return check_registry.resolve(key)


def ensure_examples(
    key: ExampleKey, *, param_name: str | None = None
) -> Tuple[ExampleValue[object], ...]:
    return example_registry.ensure(key, param_name=param_name)


def ensure_checks(key: ExampleKey, *, slot: str = "output") -> Tuple[Check, ...]:
    return check_registry.ensure(key, slot=slot)


def list_example_keys() -> Tuple[ExampleKey, ...]:
    return tuple(example_registry.iter_keys())


def list_check_keys() -> Tuple[ExampleKey, ...]:
    return tuple(check_registry.iter_keys())


def clear_registries() -> None:
    example_registry.clear()
    check_registry.clear()


__all__ = [
    "ExampleKey",
    "ExampleRegistry",
    "CheckRegistry",
    "example_registry",
    "check_registry",
    "register_example",
    "register_check",
    "resolve_examples",
    "resolve_checks",
    "ensure_examples",
    "ensure_checks",
    "list_example_keys",
    "list_check_keys",
    "clear_registries",
]
