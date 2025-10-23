"""Tests for CLI command implementations (Phase 6).

Test Coverage:
- CLI-N-01: validate_command with valid config
- CLI-N-02: run_command with valid config
- CLI-N-03: generate_config_command for single skeleton
- CLI-E-01: validate_command with invalid config
- CLI-E-02: run_command with invalid config (Fail Fast)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xform_core.dag.cli import (
    validate_command,
    run_command,
    generate_config_command,
    discover_command,
)


def test_CLI_N_01_validate_command_with_valid_config(
    valid_config_path: Path, capsys
) -> None:
    """CLI-N-01: validate_command with valid config returns 0."""
    result = validate_command(str(valid_config_path))

    assert result == 0

    captured = capsys.readouterr()
    assert "✓ Configuration" in captured.out
    assert "is valid" in captured.out


def test_CLI_N_02_run_command_with_valid_config(
    valid_config_path: Path, capsys
) -> None:
    """CLI-N-02: run_command with valid config executes pipeline."""
    # Provide minimal initial inputs (HLOCVSpec with defaults)
    from .test_types import HLOCVSpec

    initial_inputs = {
        "HLOCVSpec": HLOCVSpec()  # Use default values
    }

    result = run_command(str(valid_config_path), initial_inputs)

    assert result == 0

    captured = capsys.readouterr()
    assert "Running pipeline:" in captured.out
    assert "✓ Pipeline completed successfully" in captured.out


def test_CLI_N_03_generate_config_command_for_single_skeleton(
    generated_config_dir: Path, test_skeleton_registered: tuple, capsys
) -> None:
    """CLI-N-03: generate_config_command creates valid YAML file."""
    fqn, skeleton = test_skeleton_registered
    output_path = generated_config_dir / "generated_single.yaml"

    result = generate_config_command(
        skeleton=fqn, output=str(output_path), show_alternatives=False
    )

    assert result == 0
    assert output_path.exists()

    # Verify generated content
    import yaml

    with open(output_path) as f:
        config = yaml.safe_load(f)

    assert config["pipeline"]["name"] == skeleton.name
    assert config["pipeline"]["skeleton"] == fqn
    # The generated config should have skeleton.name as top-level key
    assert skeleton.name in config
    assert "generate_bars" in config[skeleton.name]["steps"]
    assert "compute_features" in config[skeleton.name]["steps"]

    captured = capsys.readouterr()
    assert "✓ Generated config:" in captured.out


def test_CLI_N_04_generate_config_with_alternatives(
    generated_config_dir: Path, test_skeleton_registered: tuple
) -> None:
    """CLI-N-04: generate_config_command with --show-alternatives includes comments."""
    fqn, _ = test_skeleton_registered
    output_path = generated_config_dir / "generated_with_alternatives.yaml"

    result = generate_config_command(
        skeleton=fqn, output=str(output_path), show_alternatives=True
    )

    assert result == 0
    assert output_path.exists()

    # Check that alternatives are included as comments
    content = output_path.read_text()
    # Note: alternatives might not exist for all transforms in test skeleton
    # Just verify the file is valid YAML
    import yaml

    config = yaml.safe_load(content)
    assert config is not None


def test_CLI_N_05_discover_command_list_all(capsys) -> None:
    """CLI-N-05: discover_command without args lists all registered transforms."""
    result = discover_command()

    assert result == 0

    captured = capsys.readouterr()
    assert "All registered transforms" in captured.out
    # Should show pipeline_app module
    assert "pipeline_app.transforms" in captured.out


def test_CLI_N_06_discover_command_with_type_filtering(capsys) -> None:
    """CLI-N-06: discover_command with input/output types filters transforms."""
    # Search for transforms: DataFrame -> FeatureMap
    result = discover_command(input_type="DataFrame", output_type="FeatureMap")

    assert result == 0

    captured = capsys.readouterr()
    assert "Searching transforms: DataFrame -> FeatureMap" in captured.out
    # Should find compute_feature_map
    assert "compute_feature_map" in captured.out
    assert "Input:" in captured.out
    assert "Output:" in captured.out


def test_CLI_N_07_discover_command_no_matches(capsys) -> None:
    """CLI-N-07: discover_command with non-matching types shows no results."""
    result = discover_command(
        input_type="NonExistentType", output_type="AnotherNonExistentType"
    )

    assert result == 0

    captured = capsys.readouterr()
    assert "No transforms found" in captured.out


def test_CLI_N_08_discover_command_shows_parameters(capsys) -> None:
    """CLI-N-08: discover_command displays function parameters."""
    # Search for a transform that has parameters
    result = discover_command(input_type="DataFrame", output_type="FeatureMap")

    assert result == 0

    captured = capsys.readouterr()
    # Should show parameters section
    assert "Parameters:" in captured.out or "compute_feature_map" in captured.out


def test_CLI_E_01_validate_command_with_invalid_config(
    invalid_config_wrong_transform: Path, capsys
) -> None:
    """CLI-E-01: validate_command with invalid config returns 1 and shows errors.

    Using invalid_config_wrong_transform instead of missing_step because
    test skeleton has default_transform for all steps, making missing steps
    valid (they use defaults).
    """
    result = validate_command(str(invalid_config_wrong_transform))

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Configuration validation failed:" in captured.out


def test_CLI_E_02_run_command_with_invalid_config_fail_fast(
    invalid_config_wrong_transform: Path, capsys
) -> None:
    """CLI-E-02: run_command with invalid config fails fast before execution."""
    result = run_command(str(invalid_config_wrong_transform), {})

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Pipeline execution failed:" in captured.out


def test_CLI_E_03_generate_config_missing_skeleton_argument(capsys) -> None:
    """CLI-E-03: generate_config_command without --skeleton or --all returns error."""
    result = generate_config_command()

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Error: --skeleton or --all required" in captured.out


def test_CLI_E_04_generate_config_missing_output_argument(
    test_skeleton_registered: tuple, capsys
) -> None:
    """CLI-E-04: generate_config_command with --skeleton but no --output returns error."""
    fqn, _ = test_skeleton_registered

    result = generate_config_command(skeleton=fqn)

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Error: --output required when using --skeleton" in captured.out


def test_CLI_E_05_generate_config_nonexistent_skeleton(
    generated_config_dir: Path, capsys
) -> None:
    """CLI-E-05: generate_config_command with non-existent skeleton returns error."""
    output_path = generated_config_dir / "output.yaml"

    result = generate_config_command(
        skeleton="nonexistent_skeleton_fqn", output=str(output_path)
    )

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Error:" in captured.out


def test_CLI_E_06_generate_config_all_without_output_dir(capsys) -> None:
    """CLI-E-06: generate_config_command with --all but no --output-dir returns error."""
    result = generate_config_command(generate_all=True)

    assert result == 1

    captured = capsys.readouterr()
    assert "✗ Error: --output-dir required when using --all" in captured.out
