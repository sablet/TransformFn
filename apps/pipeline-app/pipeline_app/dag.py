"""DAG description for the sample pipeline application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple, cast

from proj_dtypes.hlocv_spec import HLOCVSpec
from xform_core import TransformFn

from . import transforms


def _resolve_transform(fn: Callable[..., object]) -> TransformFn:
    """Extract the TransformFn metadata attached by the decorator."""

    transform = getattr(fn, "__transform_fn__", None)
    if transform is None:  # pragma: no cover - defensive guard
        raise AttributeError(f"function is not decorated with @transform: {fn!r}")
    return cast(TransformFn, transform)


@dataclass(frozen=True)
class Node:
    """Represents a single runnable transform inside the pipeline DAG."""

    name: str
    func: Callable[..., object]
    transform: TransformFn
    inputs: Tuple[Tuple[str, str], ...] = tuple()
    parameters: Tuple[Tuple[str, object], ...] = tuple()

    def build_kwargs(self, resolved_outputs: Dict[str, object]) -> Dict[str, object]:
        """Materialise keyword arguments from dependencies and static params."""

        kwargs: Dict[str, object] = {key: value for key, value in self.parameters}
        for param_name, dependency in self.inputs:
            kwargs[param_name] = resolved_outputs[dependency]
        return kwargs

    @property
    def dependency_names(self) -> Tuple[str, ...]:
        return tuple(dependency for _, dependency in self.inputs)


@dataclass(frozen=True)
class Pipeline:
    """A simple immutable collection of pipeline nodes."""

    nodes: Tuple[Node, ...]

    def get(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"unknown pipeline node: {name}")

    def topological_order(self) -> Tuple[Node, ...]:
        resolved: set[str] = set()
        ordered: list[Node] = []

        remaining = list(self.nodes)
        while remaining:
            progress = False
            for node in list(remaining):
                if all(dep in resolved for dep in node.dependency_names):
                    ordered.append(node)
                    resolved.add(node.name)
                    remaining.remove(node)
                    progress = True
            if not progress:
                cycle = ", ".join(node.name for node in remaining)
                raise RuntimeError(f"cyclic pipeline detected: {cycle}")
        return tuple(ordered)

    def __iter__(self) -> Iterable[Node]:
        return iter(self.nodes)


DEFAULT_PIPELINE_SPEC = HLOCVSpec(n=128, seed=99)

PIPELINE = Pipeline(
    nodes=(
        Node(
            name="price_bars",
            func=transforms.generate_price_bars,
            transform=_resolve_transform(transforms.generate_price_bars),
            parameters=(("spec", DEFAULT_PIPELINE_SPEC),),
        ),
        Node(
            name="feature_map",
            func=transforms.compute_feature_map,
            transform=_resolve_transform(transforms.compute_feature_map),
            inputs=(("bars", "price_bars"),),
            parameters=(("annualization_factor", 252.0),),
        ),
        Node(
            name="top_features",
            func=transforms.select_top_features,
            transform=_resolve_transform(transforms.select_top_features),
            inputs=(("features", "feature_map"),),
            parameters=(("top_n", 3),),
        ),
    ),
)


__all__ = [
    "DEFAULT_PIPELINE_SPEC",
    "PIPELINE",
    "Node",
    "Pipeline",
]
