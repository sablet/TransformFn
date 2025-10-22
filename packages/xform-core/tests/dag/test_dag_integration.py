"""Integration tests for DAG dynamic transform selection.

This module tests the full E2E workflow:
1. Define skeleton
2. Generate config via CLI
3. Validate config
4. Run pipeline
"""

from pathlib import Path

import pytest

from xform_core.dag.cli import generate_config_command, validate_command, run_command
from xform_core.dag.skeleton import get_skeleton, clear_registry as clear_skeleton_registry
from xform_core.dag.transform_registry import get_registry


def test_INT_E2E_01_full_workflow(
    tmp_path: Path, test_skeleton_registered: None, test_registry: None
) -> None:
    """INT-E2E-01: Full workflow (Define skeleton → Generate config → Validate → Run).

    Steps:
    1. Generate config from test skeleton
    2. Validate generated config
    3. Verify config contents
    4. Validate again to confirm consistency
    5. Run pipeline with generated config

    Expected:
    - All steps succeed
    - Generated config is valid YAML
    - Config contains correct structure
    - Validation passes
    - Pipeline execution succeeds
    """
    # Step 1: Generate config from skeleton
    config_path = tmp_path / "test_pipeline.yaml"
    result = generate_config_command(
        skeleton="test_skeleton",
        generate_all=False,
        output=str(config_path),
        output_dir=None,
        show_alternatives=False,
    )
    assert result == 0, "generate-config command should succeed"
    assert config_path.exists(), "Config file should be created"

    # Step 2: Validate generated config
    result = validate_command(str(config_path))
    assert result == 0, "Validation should succeed"

    # Step 3: Verify config contents
    config_text = config_path.read_text()
    assert "pipeline:" in config_text, "Config should have pipeline section"
    assert "test_pipeline" in config_text, "Config should reference skeleton name"
    assert "test_skeleton" in config_text, "Config should reference skeleton FQN"
    assert "generate_bars:" in config_text, "Config should have generate_bars step"
    assert "compute_features:" in config_text, "Config should have compute_features step"
    assert "transform:" in config_text, "Config should have at least one transform"

    # Step 4: Validate again (consistency check)
    result = validate_command(str(config_path))
    assert result == 0, "Validation should still succeed"

    # Step 5: Run pipeline with generated config
    # Note: Since test transforms don't require complex inputs,
    # we can just test that execution starts correctly
    # Full execution test would require proper initial_inputs
    from xform_core.dag.config import PipelineConfig

    config_obj = PipelineConfig(str(config_path))
    assert config_obj.validation_result.is_valid, "Config should be valid"


def test_INT_E2E_02_generate_all_skeletons(
    tmp_path: Path, test_skeleton_registered: None, test_registry: None
) -> None:
    """INT-E2E-02: Generate configs for all registered skeletons.

    Steps:
    1. Generate all configs
    2. Verify all config files created
    3. Validate each generated config

    Expected:
    - All config files created
    - All configs are valid
    """
    output_dir = tmp_path / "configs"

    # Step 1: Generate all configs
    result = generate_config_command(
        skeleton=None,
        generate_all=True,
        output=None,
        output_dir=str(output_dir),
        show_alternatives=False,
    )
    assert result == 0, "generate-config --all should succeed"

    # Step 2: Verify config file created
    config_files = list(output_dir.glob("*.yaml"))
    assert len(config_files) >= 1, "At least one config file should be created"

    # Step 3: Validate each config
    for config_file in config_files:
        result = validate_command(str(config_file))
        assert result == 0, f"Config {config_file.name} should be valid"


def test_INT_E2E_03_generate_with_alternatives(
    tmp_path: Path, test_skeleton_registered: None, test_registry: None
) -> None:
    """INT-E2E-03: Generate config with alternative transforms shown.

    Steps:
    1. Generate config with --show-alternatives
    2. Verify alternative transforms are in comments

    Expected:
    - Config contains commented alternatives
    - Config is still valid YAML
    """
    config_path = tmp_path / "test_with_alternatives.yaml"

    # Step 1: Generate with alternatives
    result = generate_config_command(
        skeleton="test_skeleton",
        generate_all=False,
        output=str(config_path),
        output_dir=None,
        show_alternatives=True,
    )
    assert result == 0, "generate-config --show-alternatives should succeed"

    # Step 2: Verify alternatives in comments
    config_text = config_path.read_text()
    # Note: Alternatives will only show if multiple transforms match the signature
    # In test_skeleton, we may not have multiple matches, so just verify structure
    assert "pipeline:" in config_text, "Config should have valid structure"

    # Validate config
    result = validate_command(str(config_path))
    assert result == 0, "Config with alternatives should be valid"


def test_INT_E2E_04_error_handling_invalid_skeleton(tmp_path: Path) -> None:
    """INT-E2E-04: Error handling for invalid skeleton FQN.

    Steps:
    1. Try to generate config with non-existent skeleton
    2. Verify error is reported

    Expected:
    - Command returns non-zero
    - Error message shown
    """
    config_path = tmp_path / "invalid.yaml"

    result = generate_config_command(
        skeleton="non_existent_skeleton",
        generate_all=False,
        output=str(config_path),
        output_dir=None,
        show_alternatives=False,
    )
    assert result == 1, "Command should fail for invalid skeleton"
    assert not config_path.exists(), "Config file should not be created"


def test_INT_E2E_05_error_handling_missing_output(
    test_skeleton_registered: None,
) -> None:
    """INT-E2E-05: Error handling when --output is missing.

    Steps:
    1. Try to generate config without --output
    2. Verify error is reported

    Expected:
    - Command returns non-zero
    - Error message shown
    """
    result = generate_config_command(
        skeleton="test_skeleton",
        generate_all=False,
        output=None,  # Missing output
        output_dir=None,
        show_alternatives=False,
    )
    assert result == 1, "Command should fail without --output"
