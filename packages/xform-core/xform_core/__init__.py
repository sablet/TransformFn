"""xform-core: TransformFn 正規化のコア API。"""

from .metadata import Check, ExampleType, ExampleValue
from .models import CodeRef, ParamField, ParamSchema, Schema, SchemaField, TransformFn
from .registry import (
    CheckEntry,
    CheckRegistry,
    ExampleEntry,
    ExampleRegistry,
    check_registry,
    example_registry,
)
from .transforms_core import normalize_transform, transform

__all__ = [
    "Check",
    "ExampleType",
    "ExampleValue",
    "CheckEntry",
    "CheckRegistry",
    "ExampleEntry",
    "ExampleRegistry",
    "CodeRef",
    "ParamField",
    "ParamSchema",
    "Schema",
    "SchemaField",
    "TransformFn",
    "check_registry",
    "example_registry",
    "normalize_transform",
    "transform",
]
