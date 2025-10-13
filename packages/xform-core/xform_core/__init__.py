"""xform-core: TransformFn 正規化のコア API。"""

from .exceptions import (
    MissingCheckError,
    MissingExampleError,
    RegistryNotInitializedError,
    ResolutionError,
)
from .metadata import Check, ExampleType, ExampleValue
from .models import CodeRef, ParamField, ParamSchema, Schema, SchemaField, TransformFn
from .registry import (
    CheckRegistry as TypeCheckRegistry,
    ExampleRegistry as TypeExampleRegistry,
    check_registry as type_check_registry,
    example_registry as type_example_registry,
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
from .transform_registry import (
    CheckEntry,
    CheckRegistry,
    ExampleEntry,
    ExampleRegistry,
    check_registry,
    example_registry,
)
from .transforms_core import allow_transform_errors, normalize_transform, transform

__all__ = [
    "Check",
    "ExampleType",
    "ExampleValue",
    "CheckEntry",
    "CheckRegistry",
    "ExampleEntry",
    "ExampleRegistry",
    "TypeCheckRegistry",
    "TypeExampleRegistry",
    "ResolutionError",
    "MissingExampleError",
    "MissingCheckError",
    "RegistryNotInitializedError",
    "CodeRef",
    "ParamField",
    "ParamSchema",
    "Schema",
    "SchemaField",
    "TransformFn",
    "check_registry",
    "example_registry",
    "allow_transform_errors",
    "normalize_transform",
    "transform",
    "register_example",
    "register_check",
    "resolve_examples",
    "resolve_checks",
    "ensure_examples",
    "ensure_checks",
    "list_example_keys",
    "list_check_keys",
    "clear_registries",
    "type_example_registry",
    "type_check_registry",
]
