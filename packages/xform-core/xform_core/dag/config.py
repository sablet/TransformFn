"""Configuration loading with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xform_core.dag.skeleton import get_skeleton
from xform_core.dag.transform_registry import get_registry
from xform_core.dag.validator import ConfigurationValidator


class PipelineConfig:
    """Pipeline configuration with automatic validation."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        # Load YAML
        with open(self.config_path) as f:
            self.raw = yaml.safe_load(f)

        # Get skeleton
        skeleton_fqn = self.raw.get("pipeline", {}).get("skeleton")
        if not skeleton_fqn:
            raise ValueError("Configuration missing 'pipeline.skeleton' field")

        self.skeleton = get_skeleton(skeleton_fqn)

        # Validate IMMEDIATELY on load
        registry = get_registry()
        validator = ConfigurationValidator(registry, self.skeleton)

        # Get phase-specific config for validation
        phase_config = self.raw.get(self.skeleton.name, {})
        self.validation_result = validator.validate(phase_config)

        # FAIL FAST if invalid
        if not self.validation_result.is_valid:
            raise ValueError(
                f"Configuration validation failed for {config_path}:\n"
                f"{self.validation_result}"
            )

    def get_phase_config(self, phase: str) -> dict[str, Any]:
        """Get configuration for a specific phase."""
        return self.raw.get(phase, {})


__all__ = ["PipelineConfig"]
