"""mypy 用 TransformFn ルールプラグイン。"""

from __future__ import annotations

import importlib
import os
from typing import Callable, Type

from mypy.plugin import FunctionContext, Plugin
from mypy.types import Type as MypyType

from xform_core.transforms_core import PLUGIN_ENV_FLAG


class TransformRulesPlugin(Plugin):
    """@transform 呼び出しを監視し、TR001〜TR015 を強制する。"""

    TARGETS = {
        "xform_core.transform",
        "xform_core.transforms_core.transform",
    }

    def get_function_hook(self, fullname: str):
        if fullname in self.TARGETS:
            return transform_function_hook
        return None


def transform_function_hook(ctx: FunctionContext) -> MypyType:
    arg_nodes = getattr(ctx, "arg_nodes", None)
    if not arg_nodes or not arg_nodes[0]:
        return ctx.default_return_type
    func_expr = arg_nodes[0][0]
    fullname = getattr(func_expr, "fullname", None)
    if not fullname:
        return ctx.default_return_type
    module_name, _, attr = fullname.rpartition(".")
    if not module_name or not attr:
        return ctx.default_return_type

    previous = os.environ.get(PLUGIN_ENV_FLAG)
    os.environ[PLUGIN_ENV_FLAG] = "1"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import 失敗は稀
        ctx.api.fail(f"TR000: モジュールの読み込みに失敗しました: {exc}", func_expr)
        return ctx.default_return_type
    finally:
        if previous is None:
            os.environ.pop(PLUGIN_ENV_FLAG, None)
        else:
            os.environ[PLUGIN_ENV_FLAG] = previous

    target = getattr(module, attr, None)
    if target is None:
        ctx.api.fail("TR000: 対象関数を取得できません", func_expr)
        return ctx.default_return_type

    _run_runtime_checks(target, ctx)
    return ctx.default_return_type


def _run_runtime_checks(func: Callable[..., object], ctx: FunctionContext) -> None:
    try:
        from xform_core.transforms_core import normalize_transform

        normalize_transform(func)
    except ValueError as exc:
        ctx.api.fail(str(exc), ctx.context)
    except Exception as exc:  # pragma: no cover - 想定外
        ctx.api.fail(f"TR000: 解析中に予期しない例外が発生しました: {exc}", ctx.context)


def plugin(version: str) -> Type[Plugin]:
    return TransformRulesPlugin
