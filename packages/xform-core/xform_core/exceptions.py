"""自動注釈解決に関する例外定義。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class ResolutionError(ValueError):
    """自動注釈解決に失敗した際の基底例外。"""

    key: str
    message: str
    slot: str | None = None
    param_name: str | None = None
    available_keys: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.args = (self.__str__(),)

    def __str__(self) -> str:
        base = self.message
        if self.param_name:
            base = f"{self.param_name}: {base}"
        if self.available_keys:
            joined = ", ".join(sorted(set(self.available_keys)))
            base = f"{base} (available: {joined})"
        return base


class MissingExampleError(ResolutionError):
    """ExampleValue が登録されていない場合に送出される。"""

    def __init__(
        self,
        *,
        key: str,
        param_name: str | None,
        available_keys: Sequence[str] | None,
    ) -> None:
        message = f"TR003: ExampleValue for {key} が未登録です"
        super().__init__(
            key=key,
            message=message,
            slot="input",
            param_name=param_name,
            available_keys=available_keys,
        )


class MissingCheckError(ResolutionError):
    """Check が登録されていない場合に送出される。"""

    def __init__(
        self,
        *,
        key: str,
        slot: str | None,
        available_keys: Sequence[str] | None,
    ) -> None:
        message = f"TR005: Check for {key} が未登録です"
        super().__init__(
            key=key,
            message=message,
            slot=slot or "output",
            param_name=None,
            available_keys=available_keys,
        )


class RegistryNotInitializedError(ResolutionError):
    """レジストリが初期化されていない場合に送出される。"""

    def __init__(self, *, key: str | None = None) -> None:
        message = "Example/Check registry has not been initialized"
        super().__init__(
            key=key or "<unknown>",
            message=message,
            slot=None,
            param_name=None,
            available_keys=(),
        )
