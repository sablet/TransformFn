"""Cache key generation for TransformFn reproducibility."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .models import TransformFn


def compute_cache_key(
    transform: TransformFn,
    *,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> str:
    """Build a reproducible cache key from runtime inputs and metadata.

    The cache key includes:
    - transform_id: Unique identifier for the transform function
    - transform_version: Version of the transform
    - code_hash: Hash of the function's source code
    - inputs: Hashes of input values
    - params: Hashes of parameter values
    - env_hash: Python version for environment reproducibility

    Args:
        transform: The TransformFn metadata
        inputs: Input values passed to the function
        params: Parameter values passed to the function

    Returns:
        SHA-256 hash as hexadecimal string
    """
    payload = {
        "transform_id": str(transform.id),
        "transform_version": transform.version,
        "code_hash": transform.code_ref.code_hash,
        "inputs": {key: _hash_value(value) for key, value in sorted(inputs.items())},
        "params": {key: _hash_value(value) for key, value in sorted(params.items())},
        "env_hash": platform.python_version(),
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_value(value: object) -> str:
    """Hash an arbitrary Python object for cache key generation.

    Attempts pickle serialization first for deterministic hashing.
    Falls back to repr() if pickling fails.

    Args:
        value: Any Python object

    Returns:
        SHA-256 hash as hexadecimal string
    """
    try:
        blob = pickle.dumps(value)
    except Exception:  # noqa: S110 - intentional broad catch for fallback
        blob = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "compute_cache_key",
]
