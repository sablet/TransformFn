"""Pipeline and DAG infrastructure for TransformFn execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple, cast

if TYPE_CHECKING:
    from .models import TransformFn


def _resolve_transform(fn: Callable[..., object]) -> TransformFn:
    """Extract the TransformFn metadata attached by the @transform decorator.

    Args:
        fn: Function decorated with @transform

    Returns:
        The TransformFn metadata object

    Raises:
        AttributeError: If the function is not decorated with @transform
    """
    transform = getattr(fn, "__transform_fn__", None)
    if transform is None:  # pragma: no cover - defensive guard
        msg = f"function is not decorated with @transform: {fn!r}"
        raise AttributeError(msg)
    return cast("TransformFn", transform)


@dataclass(frozen=True)
class Node:
    """Represents a single runnable transform inside the pipeline DAG.

    Attributes:
        name: Unique identifier for this node in the pipeline
        func: The callable function to execute
        transform: TransformFn metadata for cache key generation
        inputs: Parameter-to-dependency mappings (param_name, dependency_node_name)
        parameters: Static parameter-value pairs (param_name, value)
    """

    name: str
    func: Callable[..., object]
    transform: TransformFn
    inputs: Tuple[Tuple[str, str], ...] = tuple()
    parameters: Tuple[Tuple[str, object], ...] = tuple()

    def build_kwargs(self, resolved_outputs: Dict[str, object]) -> Dict[str, object]:
        """Materialise keyword arguments from dependencies and static params.

        Args:
            resolved_outputs: Dictionary of already-resolved node outputs

        Returns:
            Dictionary of keyword arguments ready for function execution
        """
        kwargs: Dict[str, object] = {key: value for key, value in self.parameters}
        for param_name, dependency in self.inputs:
            kwargs[param_name] = resolved_outputs[dependency]
        return kwargs

    @property
    def dependency_names(self) -> Tuple[str, ...]:
        """List of node names this node depends on."""
        return tuple(dependency for _, dependency in self.inputs)


@dataclass(frozen=True)
class Pipeline:
    """A simple immutable collection of pipeline nodes.

    Provides topological sorting for DAG execution and node lookup by name.

    Attributes:
        nodes: Tuple of Node objects defining the pipeline
    """

    nodes: Tuple[Node, ...]

    def get(self, name: str) -> Node:
        """Retrieve a node by name.

        Args:
            name: Node identifier

        Returns:
            The matching Node

        Raises:
            KeyError: If no node with the given name exists
        """
        for node in self.nodes:
            if node.name == name:
                return node
        msg = f"unknown pipeline node: {name}"
        raise KeyError(msg)

    def topological_order(self) -> Tuple[Node, ...]:
        """Compute topological ordering of nodes for execution.

        Returns:
            Tuple of nodes in dependency-respecting execution order

        Raises:
            RuntimeError: If the pipeline contains a cycle
        """
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
                msg = f"cyclic pipeline detected: {cycle}"
                raise RuntimeError(msg)
        return tuple(ordered)

    def __iter__(self) -> Iterable[Node]:
        """Iterate over nodes in definition order."""
        return iter(self.nodes)


__all__ = [
    "Node",
    "Pipeline",
]
