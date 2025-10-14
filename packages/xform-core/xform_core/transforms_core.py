"""TransformFn 正規化ロジックと @transform デコレータ。"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import os
from contextlib import contextmanager
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from .metadata import Check, ExampleValue, is_check_metadata, is_example_metadata
from .registry import ensure_checks, ensure_examples
from .transform_registry import (
    check_registry as transform_check_registry,
    example_registry as transform_example_registry,
)
from .models import CodeRef, ParamField, ParamSchema, Schema, SchemaField, TransformFn

TR_ERRORS = {
    "TR001": "@transform 関数には少なくとも1つの引数が必要です",
    "TR002": "最初の引数は Annotated[...] で指定してください",
    "TR003": "入力 Annotated には ExampleType または ExampleValue を含めてください",
    "TR004": "ExampleValue の型が入力の基底型と一致しません",
    "TR005": "戻り値の型は Annotated[...] で指定してください",
    "TR006": "戻り値 Annotated には Check[...] を含めてください",
    "TR007": "Check[...] の引数は文字列リテラル FQN である必要があります",
    "TR008": "Check で参照された関数が見つかりません",
    "TR009": "@transform 関数には docstring が必要です",
}

PLUGIN_ENV_FLAG = "XFORM_CORE_ALLOW_DECORATOR_ERRORS"


@contextmanager
def allow_transform_errors() -> Iterator[None]:
    """Temporarily allow @transform to attach errors instead of raising."""

    previous = os.environ.get(PLUGIN_ENV_FLAG)
    os.environ[PLUGIN_ENV_FLAG] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PLUGIN_ENV_FLAG, None)
        else:
            os.environ[PLUGIN_ENV_FLAG] = previous


def transform(
    func: Callable | None = None,
    *,
    is_pure: bool = True,
    auto_annotation: bool = True,
) -> Callable:
    """@transform デコレータ。

    デコレータ適用時に TransformFn を正規化し、
    関数へ `__transform_fn__` 属性として付与する。
    mypy プラグインがモジュールをインポートする際は環境変数でエラーを握り潰す。
    """

    if func is None:
        return lambda inner: transform(
            inner, is_pure=is_pure, auto_annotation=auto_annotation
        )

    allow_errors = os.environ.get(PLUGIN_ENV_FLAG) == "1"

    typed_func = cast(Any, func)

    try:
        transform_fn = normalize_transform(
            func, is_pure=is_pure, auto_annotation=auto_annotation
        )
    except ValueError as exc:
        if allow_errors:
            typed_func.__transform_error__ = exc
            return func
        raise
    else:
        if hasattr(func, "__transform_error__"):
            delattr(func, "__transform_error__")

    typed_func = cast(Any, func)
    typed_func.__transform_fn__ = transform_fn
    return func


def normalize_transform(
    func: Callable[..., Any], *, is_pure: bool = True, auto_annotation: bool = True
) -> TransformFn:
    _ensure_docstring(func)

    signature = inspect.signature(func)
    parameters = _ensure_parameters(signature.parameters, func)

    globalns = dict(func.__globals__)
    type_hints: Dict[str, object] = get_type_hints(
        func, globalns=globalns, localns=None, include_extras=True
    )

    first_param = parameters[0]
    base_input, input_metadata, input_examples = _extract_input_spec(
        func, first_param, type_hints, auto_annotation=auto_annotation
    )
    base_output, output_metadata, check_records = _extract_output_spec(
        func, type_hints, auto_annotation=auto_annotation
    )
    check_targets = tuple(record[0] for record in check_records)

    input_schema = _extract_schema(base_input)
    output_schema = _extract_schema(base_output)
    param_schema = _build_param_schema(parameters[1:], type_hints)

    code_ref = _build_code_ref(func)
    transform_fqn = f"{func.__module__}.{func.__qualname__}"

    transform_fn = TransformFn(
        name=func.__name__,
        qualname=func.__qualname__,
        module=func.__module__,
        input_schema=input_schema,
        output_schema=output_schema,
        param_schema=param_schema,
        code_ref=code_ref,
        engine="python",
        is_pure=is_pure,
        input_metadata=tuple(input_metadata),
        output_checks=check_targets,
    )

    example_metadata_map = _build_example_metadata_map(
        func, parameters, type_hints, input_examples, auto_annotation=auto_annotation
    )

    transform_example_registry.register_many(
        transform_fqn,
        example_metadata_map,
        source="annotation",
        overwrite=True,
    )
    transform_check_registry.register(
        transform_fqn,
        check_records,
        source="annotation",
        overwrite=True,
    )
    return transform_fn


def _ensure_docstring(func: Callable[..., Any]) -> None:
    if not inspect.getdoc(func):
        raise ValueError(_error_msg("TR009", func))


def _ensure_parameters(
    parameters: Mapping[str, inspect.Parameter], func: Callable[..., Any]
) -> list[inspect.Parameter]:
    values = list(parameters.values())
    if not values:
        raise ValueError(_error_msg("TR001", func))
    return values


def _extract_input_spec(
    func: Callable[..., Any],
    param: inspect.Parameter,
    type_hints: Mapping[str, object],
    *,
    auto_annotation: bool,
) -> tuple[object, tuple[object, ...], tuple[object, ...]]:
    annotation = type_hints.get(param.name, param.annotation)
    base_input, input_metadata, is_annotated = _split_annotation(annotation)
    if base_input is None:
        raise ValueError(_error_msg("TR002", func))

    if not is_annotated and not auto_annotation:
        raise ValueError(_error_msg("TR002", func))

    example_meta = tuple(meta for meta in input_metadata if is_example_metadata(meta))

    if not example_meta and auto_annotation:
        key = _annotation_key(base_input)
        resolved = ensure_examples(
            key,
            param_name=f"{func.__qualname__}.{param.name}",
        )
        input_metadata = _append_metadata(input_metadata, resolved)
        example_meta = tuple(resolved)

    if not example_meta:
        raise ValueError(_error_msg("TR003", func))

    _validate_example_types(base_input, example_meta, func)
    return base_input, input_metadata, example_meta


def _extract_output_spec(
    func: Callable[..., Any],
    type_hints: Mapping[str, object],
    *,
    auto_annotation: bool,
) -> tuple[object, tuple[object, ...], tuple[tuple[str, Callable[..., Any]], ...]]:
    return_annotation = type_hints.get("return")
    base_output, output_metadata, is_annotated = _split_annotation(return_annotation)
    if base_output is None:
        raise ValueError(_error_msg("TR005", func))

    if not is_annotated and not auto_annotation:
        raise ValueError(_error_msg("TR005", func))

    checks = tuple(meta for meta in output_metadata if is_check_metadata(meta))

    if not checks and auto_annotation:
        resolved_checks = _resolve_output_checks(base_output, func)
        output_metadata = _append_metadata(output_metadata, resolved_checks)
        checks = tuple(resolved_checks)

    if not checks:
        raise ValueError(_error_msg("TR006", func))

    check_records = tuple(_validate_check(meta, func) for meta in checks)
    return base_output, output_metadata, check_records


def _resolve_output_checks(
    base_output: object, func: Callable[..., Any]
) -> list[Check]:
    """Resolve Check metadata for output type, handling tuple types."""
    origin = get_origin(base_output)

    if origin is tuple:
        args = get_args(base_output)
        if not args:
            key = _annotation_key(base_output)
            return list(ensure_checks(key, slot="output"))

        element_check_funcs: list[Callable[[Any], None]] = []
        for _i, arg in enumerate(args):
            key = _annotation_key(arg)
            try:
                checks = ensure_checks(key, slot="output")
                for check_meta in checks:
                    _, check_func = _validate_check(check_meta, func)
                    element_check_funcs.append(check_func)
            except ValueError:
                continue

        if not element_check_funcs:
            key = _annotation_key(base_output)
            return list(ensure_checks(key, slot="output"))

        composite_check_func = _create_tuple_check_function(
            element_check_funcs, len(args)
        )
        check_fqn = f"{func.__module__}._auto_tuple_check_{id(func)}"
        setattr(
            __import__(func.__module__, fromlist=["_"]),
            f"_auto_tuple_check_{id(func)}",
            composite_check_func,
        )
        return [Check(check_fqn)]

    key = _annotation_key(base_output)
    return list(ensure_checks(key, slot="output"))


def _create_tuple_check_function(
    check_funcs: list[Callable[[Any], None]], expected_len: int
) -> Callable[[object], None]:
    """Create a composite check function for tuple outputs."""

    def composite_check(value: object) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"expected tuple, got {type(value).__name__}")
        if len(value) != expected_len:
            msg = f"expected tuple of length {expected_len}, got {len(value)}"
            raise ValueError(msg)
        for i, (element, check_func) in enumerate(
            zip(value, check_funcs, strict=False)
        ):
            try:
                check_func(element)
            except Exception as exc:
                raise ValueError(f"element {i} check failed: {exc}") from exc

    return composite_check


def _build_example_metadata_map(
    func: Callable[..., Any],
    parameters: Sequence[inspect.Parameter],
    type_hints: Mapping[str, object],
    first_param_examples: tuple[object, ...],
    *,
    auto_annotation: bool = True,
) -> Dict[str, tuple[object, ...]]:
    example_metadata_map: Dict[str, tuple[object, ...]] = {}
    first_param = parameters[0]
    if first_param_examples:
        example_metadata_map[first_param.name] = first_param_examples

    for param in parameters[1:]:
        annotation = type_hints.get(param.name, param.annotation)
        base_param, param_metadata, is_annotated = _split_annotation(annotation)

        if base_param is None:
            continue

        example_meta = tuple(
            meta for meta in param_metadata if is_example_metadata(meta)
        )

        if not example_meta and param.default is not inspect._empty:
            continue

        if not example_meta and auto_annotation:
            key = _annotation_key(base_param)
            resolved = ensure_examples(
                key,
                param_name=f"{func.__qualname__}.{param.name}",
            )
            param_metadata = _append_metadata(param_metadata, resolved)
            example_meta = tuple(resolved)

        if not example_meta:
            continue

        _validate_example_types(base_param, example_meta, func)
        example_metadata_map[param.name] = example_meta

    return example_metadata_map


def _split_annotation(
    annotation: object | None,
) -> tuple[object | None, tuple[object, ...], bool]:
    if annotation in (None, inspect._empty):
        return None, (), False
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if not args:
            return None, (), True
        return args[0], tuple(args[1:]), True
    return annotation, (), False


def _append_metadata(
    metadata: tuple[object, ...], additions: Sequence[object]
) -> tuple[object, ...]:
    if not additions:
        return metadata
    merged = list(metadata)
    for item in additions:
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _annotation_key(annotation: object) -> str:
    if annotation in (None, inspect._empty):
        return "<unknown>"
    try:
        from typing import ForwardRef as _ForwardRef
    except ImportError:  # pragma: no cover - ForwardRef 非搭載環境
        _ForwardRef = None  # type: ignore[assignment,misc]

    if _ForwardRef is not None and isinstance(annotation, _ForwardRef):
        return annotation.__forward_arg__

    if isinstance(annotation, type):
        mod = annotation.__module__ or "builtins"
        qualname = getattr(annotation, "__qualname__", annotation.__name__)
        return f"{mod}.{qualname}"

    origin = get_origin(annotation)
    if origin is not None:
        return _annotation_key(origin)

    mod_name: str | None = getattr(annotation, "__module__", None)
    name = getattr(annotation, "__qualname__", None) or getattr(
        annotation, "__name__", None
    )
    if mod_name and name:
        return f"{mod_name}.{name}"
    return repr(annotation)


def _error_msg(code: str, func: Callable[..., Any]) -> str:
    base = TR_ERRORS.get(code, code)
    return f"{code}: {base} (function: {func.__module__}.{func.__qualname__})"


# rest unchanged...


def _unwrap_annotated(
    annotation: object | None,
) -> tuple[object | None, tuple[object, ...]]:
    if annotation is None:
        return None, ()
    origin = get_origin(annotation)
    if origin is not Annotated:
        return None, ()
    args = get_args(annotation)
    if not args:
        return None, ()
    return args[0], tuple(args[1:])


def _validate_example_types(
    base: object, metadata: Iterable[object], func: Callable[..., Any]
) -> None:
    for meta in metadata:
        if isinstance(meta, ExampleValue):
            if not _example_matches(base, meta.value):
                raise ValueError(_error_msg("TR004", func))


def _example_matches(base: object, value: object) -> bool:
    if value is None:
        return True
    if isinstance(base, type):
        try:
            return isinstance(value, base)
        except TypeError:
            return True
    return True


def _validate_check(
    meta: object, func: Callable[..., Any]
) -> tuple[str, Callable[..., Any]]:
    if not isinstance(meta, Check):
        raise ValueError(_error_msg("TR006", func))
    target = meta.target
    if not isinstance(target, str) or not target:
        raise ValueError(_error_msg("TR007", func))
    module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        raise ValueError(_error_msg("TR007", func))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(_error_msg("TR008", func)) from exc
    obj = getattr(module, attr, None)
    if obj is None or not callable(obj):
        raise ValueError(_error_msg("TR008", func))
    return target, obj


def _extract_schema(annotation_base: object) -> Schema:
    if annotation_base is None:
        return Schema(name="Unknown", fields=tuple())

    if isinstance(annotation_base, type):
        if is_typeddict(annotation_base):
            fields = _schema_from_typeddict(annotation_base)
            return Schema(name=annotation_base.__name__, fields=fields)
        if dataclasses.is_dataclass(annotation_base):
            fields = _schema_from_dataclass(annotation_base)
            return Schema(name=annotation_base.__name__, fields=fields)
        return Schema(name=annotation_base.__name__, fields=tuple())

    if dataclasses.is_dataclass(annotation_base):
        tp = type(annotation_base)
        fields = _schema_from_dataclass(tp)
        return Schema(name=tp.__name__, fields=fields)

    return Schema(name=repr(annotation_base), fields=tuple())


def _schema_from_typeddict(tp: type) -> tuple[SchemaField, ...]:
    annotations: dict[str, object] = getattr(tp, "__annotations__", {})
    required_keys: set[str] | frozenset[str] = getattr(
        tp, "__required_keys__", frozenset()
    )
    fields = []
    for name, annotation in annotations.items():
        dtype = _type_repr(annotation)
        nullable = name not in required_keys
        fields.append(SchemaField(name=name, dtype=dtype, nullable=nullable))
    return tuple(fields)


def _schema_from_dataclass(tp: type) -> tuple[SchemaField, ...]:
    hints = get_type_hints(tp, include_extras=True)
    fields = []
    for field in dataclasses.fields(tp):
        annotation = hints.get(field.name, field.type)
        dtype = _type_repr(annotation)
        nullable = _is_optional(annotation)
        description = field.metadata.get("description") if field.metadata else None
        fields.append(
            SchemaField(
                name=field.name, dtype=dtype, nullable=nullable, description=description
            )
        )
    return tuple(fields)


def _build_param_schema(
    params: Sequence[inspect.Parameter], type_hints: Mapping[str, object]
) -> ParamSchema:
    param_fields = []
    for param in params:
        annotation = type_hints.get(param.name, param.annotation)
        dtype = _type_repr(annotation)
        required = param.default is inspect._empty
        default = None if required else param.default
        param_fields.append(
            ParamField(name=param.name, dtype=dtype, required=required, default=default)
        )
    return ParamSchema(params=tuple(param_fields))


def _build_code_ref(func: Callable[..., Any]) -> CodeRef:
    try:
        source = inspect.getsource(func)
    except OSError:
        source = repr(func.__code__.co_code)
    code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    try:
        filepath = inspect.getsourcefile(func)
        _, lineno = inspect.getsourcelines(func)
    except OSError:
        filepath = None
        lineno = None
    return CodeRef(
        module=func.__module__,
        qualname=func.__qualname__,
        filepath=filepath,
        lineno=lineno,
        code_hash=code_hash,
    )


def _type_repr(annotation: object) -> str:
    if annotation in (None, inspect._empty):
        return "Any"
    if Annotated is not None and get_origin(annotation) is Annotated:
        return _type_repr(get_args(annotation)[0])
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation)


def _is_optional(annotation: object) -> bool:
    if annotation is None:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = get_args(annotation)
    if origin is Union:
        return any(arg is type(None) for arg in args)
    if origin is UnionType:
        return any(arg is type(None) for arg in args)
    return False
