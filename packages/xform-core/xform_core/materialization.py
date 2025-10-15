"""Example値を具象化するMaterializerプロトコル。"""

from __future__ import annotations

from typing import Protocol, TypeVar

import pandas as pd

T_contra = TypeVar("T_contra", contravariant=True)


class Materializer(Protocol[T_contra]):
    """宣言的な仕様から実行時オブジェクトを構築する契約。"""

    def materialize(self, spec: T_contra) -> object:
        """仕様オブジェクトを具象値へ変換する。"""
        ...


def default_materializer(value: object) -> object:
    """共通Materializer: DataFrameは防御的コピー、それ以外は素通し。"""
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return value


__all__ = ["Materializer", "default_materializer"]
