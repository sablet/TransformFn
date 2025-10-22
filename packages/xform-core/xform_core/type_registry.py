"""Registered schema/container type registry.

This module keeps track of the types that can be used as payload schemas for
``@transform`` functions.  Transform inputs/outputs must always reference a
RegisteredType (struct) or a RegisteredContainer (container schema).
"""

from __future__ import annotations

import inspect
import sys
import types
import warnings
import enum
from dataclasses import dataclass, is_dataclass
from threading import RLock
from typing import Any, Dict, Mapping
from typing import ForwardRef, Literal, get_origin

try:  # Python 3.11+
    from typing import is_typeddict as _is_typeddict
except ImportError:  # pragma: no cover - fallback for older runtimes
    from typing_extensions import is_typeddict as _is_typeddict  # type: ignore

try:  # Optional dependency; only used for isinstance checks if available
    from pydantic import BaseModel as _BaseModel  # type: ignore
except Exception:  # pragma: no cover - pydantic 未インストール時
    _BaseModel = None


class TypeRegistryError(ValueError):
    """Raised when a type registration rule is violated."""


def _default_container_alias(container: str, value_key: str) -> str:
    value_name = value_key.rsplit(".", 1)[-1]
    return f"{container}Of{value_name}"


def _inspect_module_name(default: str | None = None) -> str:
    frame = inspect.currentframe()
    if frame is None:
        return default or "__main__"
    caller = frame.f_back
    if caller is None:
        return default or "__main__"
    module_name = caller.f_globals.get("__name__", None)
    if not isinstance(module_name, str):
        return default or "__main__"
    return module_name


def annotation_key(annotation: object) -> str:
    """Return the normalized key used across registries."""

    if annotation in (None, inspect._empty):
        raise TypeRegistryError("annotation cannot be empty for key computation")

    if isinstance(annotation, ForwardRef):
        return annotation.__forward_arg__

    origin = get_origin(annotation)
    if origin is not None and origin is not annotation:
        return annotation_key(origin)

    module_name = getattr(annotation, "__module__", None)
    qual = getattr(annotation, "__qualname__", None)
    result: str | None = None

    if isinstance(annotation, str):
        result = annotation
    elif isinstance(annotation, type):
        module = annotation.__module__ or "builtins"
        qualname = getattr(annotation, "__qualname__", annotation.__name__)
        result = f"{module}.{qualname}"
    elif module_name and qual:
        result = f"{module_name}.{qual}"
    else:
        name = getattr(annotation, "__name__", None)
        if name:
            module_part = module_name or "builtins"
            result = f"{module_part}.{name}"

    if result is None:
        result = repr(annotation)

    return result


def _is_allowed_struct_type(type_ref: object) -> bool:
    """Return True if the type can be registered as a struct payload."""

    if not isinstance(type_ref, type):
        return False

    if _is_typeddict(type_ref):
        return True

    if is_dataclass(type_ref):
        return True

    if issubclass(type_ref, enum.Enum):
        return True

    if _BaseModel is not None and issubclass(type_ref, _BaseModel):
        return True

    if type_ref in (int, float, bool, str, list, tuple, dict, set):
        return True

    return False


def _has_named_repo_alias(type_ref: type, caller_module: str) -> bool:
    """Return True if caller module exposes a renamed alias to type_ref."""

    if not _is_allowed_module_prefix(caller_module):
        return False

    target_module = getattr(type_ref, "__module__", "")
    if _is_allowed_module_prefix(target_module):
        return False

    module = sys.modules.get(caller_module)
    if module is None:
        return False

    original_names = {
        getattr(type_ref, "__name__", ""),
        getattr(type_ref, "__qualname__", ""),
    }

    for attr_name, value in vars(module).items():
        if value is type_ref and attr_name not in original_names:
            return True

    return False


def _is_allowed_module_prefix(module_name: str) -> bool:
    return any(module_name.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES)


@dataclass(frozen=True)
class ContainerSpec:
    """Metadata attached to a registered container schema."""

    container: str
    value_key: str
    key_key: str | None = None
    metadata: Mapping[str, Any] = types.MappingProxyType({})


class _TypeRegistry:
    """Thread-safe container for registered schema/container types."""

    def __init__(self) -> None:
        self._structs: set[str] = set()
        self._blocked_structs: set[str] = set()
        self._containers: Dict[str, ContainerSpec] = {}
        self._lock = RLock()

    def register_struct(self, key: str, *, blocked: bool = False) -> None:
        with self._lock:
            self._structs.add(key)
            if blocked:
                self._blocked_structs.add(key)
            else:
                self._blocked_structs.discard(key)

    def register_container(self, key: str, spec: ContainerSpec) -> None:
        with self._lock:
            existing = self._containers.get(key)
            if existing is not None and existing != spec:
                raise TypeRegistryError(
                    f"container schema already registered for {key}: {existing}"
                )
            self._containers[key] = spec

    def is_struct(self, key: str) -> bool:
        with self._lock:
            return key in self._structs

    def get_container(self, key: str) -> ContainerSpec | None:
        with self._lock:
            return self._containers.get(key)

    def is_registered(self, key: str) -> bool:
        with self._lock:
            return key in self._structs or key in self._containers

    def clear(self) -> None:
        with self._lock:
            self._structs.clear()
            self._blocked_structs.clear()
            self._containers.clear()

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return key in self._blocked_structs


_REGISTRY = _TypeRegistry()

_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "apps.",
    "packages.",
    "components.",
    "src.",
    "__main__",
    "tests.",
)

AllowedContainer = Literal["List", "DictStrKeys", "MappingStrKeys"]
_ALLOWED_CONTAINERS: Dict[str, Dict[str, Any]] = {
    "List": {"requires_key": False, "key_choices": ()},
    "DictStrKeys": {
        "requires_key": True,
        "key_choices": ("builtins.str",),
    },
    "MappingStrKeys": {
        "requires_key": True,
        "key_choices": ("builtins.str",),
    },
}


def clear_registered_types() -> None:
    """Reset type registry (useful for tests)."""

    _REGISTRY.clear()


def register_struct_type(type_ref: object) -> str:
    """Mark a type as a registered payload schema."""

    caller_module = _inspect_module_name()

    if isinstance(type_ref, str):
        key = type_ref
        module_name, _, _ = key.rpartition(".")
        is_allowed = _is_allowed_module_prefix(module_name) or module_name == "builtins"
        blocked = not is_allowed
        if blocked:
            warnings.warn(
                (
                    f"registering third-party schema {key}; "
                    "define a project-specific alias or structured type"
                ),
                stacklevel=2,
            )
        _REGISTRY.register_struct(key, blocked=blocked)
        return key

    key = annotation_key(type_ref)
    is_allowed = _is_allowed_struct_type(type_ref)
    has_alias = _has_named_repo_alias(type_ref, caller_module)
    blocked = not (is_allowed or has_alias)

    if blocked:
        module_name = getattr(type_ref, "__module__", "unknown")
        warnings.warn(
            (
                f"registering third-party type {key} from {module_name}; "
                "define a project-specific alias or structured schema to avoid TR010"
            ),
            stacklevel=2,
        )

    _REGISTRY.register_struct(key, blocked=blocked)
    return key


def register_container_type(
    alias: object,
    *,
    container: AllowedContainer,
    value_type: object,
    key_type: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Register a container schema alias."""

    if container not in _ALLOWED_CONTAINERS:
        allowed = ", ".join(sorted(_ALLOWED_CONTAINERS))
        raise TypeRegistryError(
            f"unsupported container type {container!r}; allow {allowed}"
        )

    alias_key = annotation_key(alias)
    value_key = annotation_key(value_type)
    if not _REGISTRY.is_registered(value_key):
        raise TypeRegistryError(
            f"value type {value_key!r} must be registered before container schemas"
        )

    key_key: str | None = None
    container_rule = _ALLOWED_CONTAINERS[container]
    if container_rule["requires_key"]:
        if key_type is None:
            raise TypeRegistryError(
                f"container {container} requires key type; pass key_type argument"
            )
        key_key = annotation_key(key_type)
        if key_key not in container_rule["key_choices"]:
            allowed_keys = ", ".join(container_rule["key_choices"])
            raise TypeRegistryError(
                f"container {container} allows key types {allowed_keys}; got {key_key}"
            )
    elif key_type is not None:
        raise TypeRegistryError(
            f"container {container} does not accept key_type argument"
        )

    spec = ContainerSpec(
        container=container,
        value_key=value_key,
        key_key=key_key,
        metadata=types.MappingProxyType(dict(metadata or {})),
    )
    _REGISTRY.register_container(alias_key, spec)
    return alias_key


def is_registered_schema(annotation: object) -> bool:
    """Return True if the annotation represents a registered struct or container."""

    try:
        key = annotation_key(annotation)
    except TypeRegistryError:
        return False
    return _REGISTRY.is_registered(key)


def resolve_container(annotation: object) -> ContainerSpec | None:
    """Return container metadata if annotation is a registered container."""

    try:
        key = annotation_key(annotation)
    except TypeRegistryError:
        return None
    return _REGISTRY.get_container(key)


def ensure_registered(annotation: object, *, context: str) -> None:
    """Ensure the annotation corresponds to a registered schema."""

    key = annotation_key(annotation)
    if not _REGISTRY.is_registered(key):
        raise TypeRegistryError(f"{context}: {key} is not a registered schema type")
    if _REGISTRY.is_blocked(key):
        raise TypeRegistryError(
            f"{context}: {key} refers to third-party schema; register a structured alias"
        )


class RegisteredContainerBuilder:
    """Fluent builder returned by ``RegisteredContainer[...]`` access."""

    def __init__(
        self,
        container: AllowedContainer,
        value_type: object,
        key_type: object | None,
    ) -> None:
        self._container = container
        self._value_type = value_type
        self._key_type = key_type

    def _default_alias_name(self) -> str:
        value_key = annotation_key(self._value_type)
        return _default_container_alias(self._container, value_key)

    def alias(
        self,
        name: str | None = None,
        *,
        module: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> type:
        """Create and register a new alias class for this container schema."""

        alias_name = name or self._default_alias_name()
        module_name = module or _inspect_module_name()
        alias_type = types.new_class(alias_name, ())
        alias_type.__module__ = module_name
        register_container_type(
            alias_type,
            container=self._container,
            value_type=self._value_type,
            key_type=self._key_type,
            metadata=metadata,
        )
        return alias_type

    def register(
        self,
        alias: object,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> object:
        """Register an existing alias type."""

        register_container_type(
            alias,
            container=self._container,
            value_type=self._value_type,
            key_type=self._key_type,
            metadata=metadata,
        )
        return alias

    __call__ = alias


_TWO_PARAM_ARITY = 2
_THREE_PARAM_ARITY = 3


class RegisteredContainer:
    """Entry point for container registration builder.

    Usage:

        PortfolioAllocations = RegisteredContainer["DictStrKeys", AllocationSpec](
            "PortfolioAllocations"
        )
    """

    def __class_getitem__(
        cls,
        params: object,
    ) -> RegisteredContainerBuilder:
        if not isinstance(params, tuple):
            params = (params,)

        if len(params) == _TWO_PARAM_ARITY:
            container, value_type = params
            key_type = None
        elif len(params) == _THREE_PARAM_ARITY:
            container, key_type, value_type = params
        else:
            raise TypeRegistryError(
                "RegisteredContainer[...] expects (container, value_type) or "
                "(container, key_type, value_type)"
            )

        if not isinstance(container, str):
            raise TypeRegistryError("container identifier must be string literal")

        allowed: AllowedContainer
        try:
            allowed = container  # type: ignore[assignment]
            if allowed not in _ALLOWED_CONTAINERS:
                raise KeyError(container)
        except KeyError as exc:
            available = ", ".join(sorted(_ALLOWED_CONTAINERS))
            raise TypeRegistryError(
                f"unknown container {container!r}; choose from {available}"
            ) from exc

        meta = _ALLOWED_CONTAINERS[allowed]
        if meta["requires_key"] and key_type is None:
            key_type = str

        return RegisteredContainerBuilder(
            allowed,
            value_type=value_type,
            key_type=key_type,
        )


__all__ = [
    "AllowedContainer",
    "ContainerSpec",
    "RegisteredContainer",
    "RegisteredContainerBuilder",
    "TypeRegistryError",
    "annotation_key",
    "clear_registered_types",
    "ensure_registered",
    "is_registered_schema",
    "register_container_type",
    "register_struct_type",
    "resolve_container",
]


for _builtin_type in (int, float, bool, str):
    register_struct_type(_builtin_type)
