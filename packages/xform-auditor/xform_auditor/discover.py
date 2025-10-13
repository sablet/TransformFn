"""Transform 関数探索ロジック。"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Sequence, Tuple, cast

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from xform_core.models import TransformFn


@dataclass(frozen=True)
class TransformHandle:
    """監査対象の Transform 関数参照。"""

    fqn: str
    func: Callable[..., object]
    transform: "TransformFn"


class DiscoveryError(RuntimeError):
    """モジュール探索に失敗した際の例外。"""


_ModuleSet = Dict[str, ModuleType]


def discover_transforms(targets: Sequence[str]) -> Tuple[TransformHandle, ...]:
    """与えられたターゲット群から Transform 関数を列挙する。"""

    modules: _ModuleSet = {}
    for target in targets:
        for module in _import_target(target):
            modules.setdefault(module.__name__, module)

    handles: Dict[str, TransformHandle] = {}
    for module in modules.values():
        for handle in _collect_transforms(module):
            handles.setdefault(handle.fqn, handle)
    return tuple(sorted(handles.values(), key=lambda handle: handle.fqn))


def _import_target(target: str) -> List[ModuleType]:
    """ターゲット文字列をモジュール群へ解決する。"""

    try:
        return _import_module_and_submodules(target)
    except ModuleNotFoundError:
        path = Path(target).resolve()
        if not path.exists():
            raise DiscoveryError(f"target not found: {target}") from None
        return _import_from_path(path)


def _import_module_and_submodules(module_name: str) -> List[ModuleType]:
    module = importlib.import_module(module_name)
    modules = [module]
    if getattr(module, "__path__", None):
        prefix = f"{module.__name__}."
        for _finder, name, _is_pkg in pkgutil.walk_packages(module.__path__, prefix):
            submodule = importlib.import_module(name)
            modules.append(submodule)
    return modules


def _import_from_path(path: Path) -> List[ModuleType]:
    if path.is_dir():
        if (path / "__init__.py").exists():
            module_name, sys_root = _module_name_from_package_dir(path)
            _ensure_sys_path(sys_root)
            return _import_module_and_submodules(module_name)
        modules: List[ModuleType] = []
        for child in sorted(path.iterdir()):
            if child.name.startswith("__pycache__"):
                continue
            if child.is_dir() or child.suffix == ".py":
                modules.extend(_import_from_path(child))
        return modules

    if path.suffix != ".py":
        raise DiscoveryError(f"only Python modules are supported: {path}")

    module_name, sys_root = _module_name_from_file(path)
    _ensure_sys_path(sys_root)
    return _import_module_and_submodules(module_name)


def _module_name_from_package_dir(path: Path) -> Tuple[str, Path]:
    parts = [path.name]
    current = path
    parent = current.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        current = parent
        parent = current.parent
    module_name = ".".join(reversed(parts))
    return module_name, parent


def _module_name_from_file(path: Path) -> Tuple[str, Path]:
    if path.name == "__init__.py":
        return _module_name_from_package_dir(path.parent)

    parent = path.parent
    if (parent / "__init__.py").exists():
        package_name, sys_root = _module_name_from_package_dir(parent)
        return f"{package_name}.{path.stem}", sys_root
    return path.stem, parent


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path)
    if not path_str:
        return
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _collect_transforms(module: ModuleType) -> Tuple[TransformHandle, ...]:
    handles: List[TransformHandle] = []
    seen: set[str] = set()
    stack: List[object] = [
        member for _, member in inspect.getmembers(module) if member is not module
    ]

    while stack:
        obj = stack.pop()
        if inspect.isclass(obj) and getattr(obj, "__module__", None) == module.__name__:
            stack.extend(member for _, member in inspect.getmembers(obj))

        transform = getattr(obj, "__transform_fn__", None)
        if transform is None or not callable(obj):
            continue
        module_name = getattr(obj, "__module__", None)
        if module_name != module.__name__:
            continue
        qualname = getattr(obj, "__qualname__", None)
        if not isinstance(qualname, str):
            continue
        callable_obj = cast(Callable[..., object], obj)
        fqn = f"{module_name}.{qualname}"
        if fqn in seen:
            continue
        seen.add(fqn)
        handles.append(TransformHandle(fqn=fqn, func=callable_obj, transform=transform))

    return tuple(handles)
