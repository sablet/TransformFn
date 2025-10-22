"""xform-core: TransformFn 正規化のコア API。"""

from .artifact import ArtifactRecord, ArtifactStore, PipelineRunResult
from .cache import compute_cache_key
from .checks import (
    check_column_dtype,
    check_column_monotonic,
    check_column_nonnegative,
    check_column_positive,
    check_dataframe_has_columns,
    check_dataframe_not_empty,
    check_dataframe_notnull,
)
from .exceptions import (
    MissingCheckError,
    MissingExampleError,
    RegistryNotInitializedError,
    ResolutionError,
)
from .materialization import Materializer, default_materializer
from .metadata import Check, ExampleType, ExampleValue
from .models import CodeRef, ParamField, ParamSchema, Schema, SchemaField, TransformFn
from .pipeline import Node, Pipeline
from .runner import PipelineRunner
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
from .type_metadata import RegisteredType, make_example
from .type_registry import RegisteredContainer
from .transforms_core import allow_transform_errors, normalize_transform, transform

__all__ = [
    "Check",
    "ExampleType",
    "ExampleValue",
    "RegisteredType",
    "RegisteredContainer",
    "make_example",
    "Materializer",
    "default_materializer",
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
    "compute_cache_key",
    "Node",
    "Pipeline",
    "ArtifactRecord",
    "ArtifactStore",
    "PipelineRunResult",
    "PipelineRunner",
    "check_dataframe_not_empty",
    "check_dataframe_has_columns",
    "check_dataframe_notnull",
    "check_column_monotonic",
    "check_column_dtype",
    "check_column_positive",
    "check_column_nonnegative",
]
