# DAG Dynamic Transform Selection - Test Specification (Minimal)

## 概要

dag-dynamic-transform-selection.md で定義された各コンポーネントの単体テスト項目を**最小限**に列挙。

### 全体方針
- **Configuration Validator に集中**: 全エラータイプ（8種類）を詳細テスト
- **他は最小限**: 統合型正常系 + 代表的な異常系のみ
- **合計39テスト**: 100ケースから60%削減
- **pipeline-app 活用**: Mock不要、既存の `apps/pipeline-app` を完全活用
- **generate-config 活用**: 手動YAML不要、CLI コマンドで設定生成

## テスト方針

### 原則
1. **Configuration Validator 重点化**: 全エラータイプ（8種類）+ Suggestion品質を個別に詳細テスト
2. **他コンポーネントは最小限**: 統合型正常系1ケース + 代表的な異常系1-2ケース
3. **統合型テスト**: 複数メソッドを1つのテストで検証（例: Register → Find → Get → Validate）
4. **Fail Fast**: 実行前バリデーションの動作を複数箇所で確認（Executor, Config, CLI）
5. **型安全性**: 型シグネチャ検証は Configuration Validator で集中テスト
6. **pipeline-app 完全活用**: Mock不要、実際の Transform・型・Check を使用

### テストデータ戦略
- **Mock不要**: 既存の `apps/pipeline-app` を完全活用
- **実際の Transform**: 6つの実装で多様なシグネチャパターンをカバー
- **実際の型**: `FeatureMap`, `HLOCVSpec`, `pd.DataFrame` など
- **手動YAML不要**: `generate-config` CLI で設定ファイル自動生成
- **独立性**: 各テストは独立して実行可能（共有状態なし）
- **自動登録**: `import pipeline_app.transforms` で Registry に自動登録

---

## 1. Enhanced Transform Registry (Minimal)

**Location**: `packages/xform-core/tests/test_transform_registry_enhanced.py`

### 1.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TR-N-01 | Register → Find → Get → Validate signature | Full workflow success |
| TR-E-01 | Get transform with non-existent FQN | KeyError or ValueError |
| TR-E-02 | Validate signature with type mismatch | Returns False |

**Note**: TR-N-01 は register, find_transforms, get_transform, validate_signature の4つのメソッドを1つのテストで検証。

---

## 2. DAG Skeleton Definition (Minimal)

**Location**: `packages/xform-core/tests/test_dag_skeleton.py`

### 2.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| SK-N-01 | Create Skeleton → Register → Get → Validate config | Full workflow success |
| SK-E-01 | Get skeleton with non-existent FQN | ValueError: "Skeleton not found" |

**Note**: SK-N-01 は PipelineStep/PipelineSkeleton 作成、register_skeleton, get_skeleton, validate_config を1つのテストで検証。

---

## 3. Configuration Validator (CRITICAL)

**Location**: `packages/xform-core/tests/test_configuration_validator.py`

### 3.1 正常系テスト

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CV-N-01 | Validate complete valid configuration | ValidationResult(is_valid=True, errors=[]) |
| CV-N-02 | Validate config using default transforms | ValidationResult(is_valid=True) |
| CV-N-03 | Validate config with all optional parameters | ValidationResult(is_valid=True) |
| CV-N-04 | Validate config with correct type annotations | ValidationResult(is_valid=True) |

### 3.2 異常系テスト（各エラータイプを個別に識別）

| Test ID | Test Case | Expected Error Type | Expected Message Pattern |
|---------|-----------|---------------------|--------------------------|
| CV-E-01 | Config missing required step (no default) | MISSING_REQUIRED_STEP | "Required step '...' not found" |
| CV-E-02 | Config with no transform and no default | NO_TRANSFORM_SPECIFIED | "No transform specified" |
| CV-E-03 | Config with non-existent transform FQN | TRANSFORM_NOT_FOUND | "Transform '...' not found in registry" |
| CV-E-04 | Config with mismatched input types | TYPE_SIGNATURE_MISMATCH | "Expected: (...) -> ..., Actual: ..." |
| CV-E-05 | Config with mismatched output type | TYPE_SIGNATURE_MISMATCH | "Expected: ... -> ..., Actual: ..." |
| CV-E-06 | Config with unknown parameter | UNKNOWN_PARAMETER | "Parameter '...' not accepted" |
| CV-E-07 | Config missing required parameter | MISSING_REQUIRED_PARAMETER | "Required parameter '...' not provided" |
| CV-E-08 | Config with wrong parameter type | PARAMETER_TYPE_MISMATCH | "Expected: ..., Got: ..." |

### 3.3 警告系テスト

| Test ID | Test Case | Expected Warning Type | Expected Message Pattern |
|---------|-----------|----------------------|--------------------------|
| CV-W-01 | Config with unknown step | UNKNOWN_STEP | "Step '...' not defined in skeleton" |

### 3.4 複合エラーテスト

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CV-M-01 | Config with multiple errors (2+ types) | ValidationResult with multiple errors in list |
| CV-M-02 | Config with errors and warnings | Both errors and warnings populated |

### 3.5 Suggestion Quality Tests

| Test ID | Test Case | Expected Behavior |
|---------|-----------|-------------------|
| CV-S-01 | TRANSFORM_NOT_FOUND error includes suggestions | Suggestion lists available compatible transforms |
| CV-S-02 | UNKNOWN_PARAMETER error includes valid params | Suggestion lists valid parameter names |
| CV-S-03 | TYPE_SIGNATURE_MISMATCH includes alternatives | Suggestion lists transforms with matching signature |

---

## 4. Transform Resolver (Minimal)

**Location**: `packages/xform-core/tests/test_transform_resolver.py`

### 4.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| RS-N-01 | Resolve with explicit transform + params | Returns (func, params) |
| RS-N-02 | Resolve using default transform | Returns (default_func, params) |
| RS-E-01 | Resolve with no transform and no default | ValueError: "No transform specified" |

---

## 5. DAG Executor (Minimal)

**Location**: `packages/xform-core/tests/test_dag_executor.py`

### 5.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| EX-N-01 | Execute multi-step pipeline with validation | Returns all step outputs, context propagated |
| EX-E-01 | Execute with invalid config (validation enabled) | ValueError before execution starts (Fail Fast) |
| EX-E-02 | Execute with missing required input in context | RuntimeError: "Required input type ... not found" |

**Note**: EX-N-01 は execute, context管理, step依存関係を1つのテストで検証。

---

## 6. CLI Commands (Minimal)

**Location**: `packages/xform-core/tests/test_dag_cli.py`

### 6.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CLI-N-01 | validate_command with valid config | Returns 0 |
| CLI-N-02 | run_command with valid config | Returns 0, pipeline completes |
| CLI-N-03 | generate_config_command for single skeleton | Returns 0, creates YAML file |
| CLI-E-01 | validate_command with invalid config | Returns 1, prints errors |
| CLI-E-02 | run_command with invalid config | Returns 1, validation fails (Fail Fast) |

**Note**: discover_command は実装後に必要なら追加。基本機能優先。

---

## 7. Unified CLI Entry Point (Minimal)

**Location**: `packages/xform-core/tests/test_dag_main.py`

### 7.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| MAIN-N-01 | Run with valid app_path → module import → command execution | Success, skeleton auto-registered |
| MAIN-E-01 | Run with non-existent app_path | Returns 1, prints import error |

**Note**: MAIN-N-01 は app module resolution, skeleton import, command dispatch を1つのテストで検証。

---

## 8. Configuration Loading (Minimal)

**Location**: `packages/xform-core/tests/test_pipeline_config.py`

### 8.1 必須テスト（最小限）

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CFG-N-01 | Load valid YAML → auto-validate → get phase config | Success, config accessible |
| CFG-E-01 | Load config with validation errors | ValueError raised immediately (Fail Fast) |
| CFG-E-02 | Load non-existent config file | FileNotFoundError |

**Note**: CFG-N-01 は load, validation, get_phase_config を1つのテストで検証。

---

## 9. Integration Tests (Minimal)

**Location**: `packages/xform-core/tests/test_dag_integration.py`

### 9.1 必須テスト（最小限）

| Test ID | Test Case | Validates |
|---------|-----------|-----------|
| INT-E2E-01 | Define skeleton → Generate config (CLI) → Validate (CLI) → Run (CLI) | Full workflow success |

### 9.2 Integration Test Implementation

```python
"""E2E integration test using generate-config CLI."""

import pytest
import subprocess
from pathlib import Path

def test_int_e2e_01_full_workflow_with_generated_config(
    tmp_path,
    test_skeleton_registered,
    sample_hlocv_spec,
):
    """
    Full E2E workflow:
    1. Register test skeleton (fixture)
    2. Generate config using CLI
    3. Validate config using CLI
    4. Run pipeline using CLI
    
    This test validates:
    - Skeleton registration
    - generate-config command
    - validate command
    - run command
    - Full DAG execution with real transforms
    """
    config_path = tmp_path / "generated_pipeline.yaml"
    
    # Step 1: Generate config from skeleton
    generate_result = subprocess.run([
        "uv", "run", "python", "-m", "xform_core.dag",
        "apps/pipeline-app",
        "generate-config",
        "--skeleton", "test_pipeline_skeleton",
        "--output", str(config_path),
    ], capture_output=True, text=True)
    
    assert generate_result.returncode == 0, (
        f"generate-config failed:\n"
        f"stdout: {generate_result.stdout}\n"
        f"stderr: {generate_result.stderr}"
    )
    assert config_path.exists(), "Config file not generated"
    
    # Verify generated config content
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert "pipeline" in config
    assert config["pipeline"]["skeleton"] == "test_pipeline_skeleton"
    assert "test_pipeline" in config
    assert "steps" in config["test_pipeline"]
    
    # Step 2: Validate generated config
    validate_result = subprocess.run([
        "uv", "run", "python", "-m", "xform_core.dag",
        "apps/pipeline-app",
        "validate",
        str(config_path),
    ], capture_output=True, text=True)
    
    assert validate_result.returncode == 0, (
        f"validate failed:\n"
        f"stdout: {validate_result.stdout}\n"
        f"stderr: {validate_result.stderr}"
    )
    assert "✓" in validate_result.stdout or "valid" in validate_result.stdout.lower()
    
    # Step 3: Run pipeline with generated config
    # Note: This requires proper input data setup in the implementation
    run_result = subprocess.run([
        "uv", "run", "python", "-m", "xform_core.dag",
        "apps/pipeline-app",
        "run",
        str(config_path),
    ], capture_output=True, text=True)
    
    assert run_result.returncode == 0, (
        f"run failed:\n"
        f"stdout: {run_result.stdout}\n"
        f"stderr: {run_result.stderr}"
    )
    assert "✓" in run_result.stdout or "completed" in run_result.stdout.lower()
    
    print(f"✓ Full E2E workflow completed successfully")
    print(f"  - Generated config: {config_path}")
    print(f"  - Validation: PASSED")
    print(f"  - Execution: PASSED")
```

**Note**: 
- INT-E2E-01 が成功すれば、全コンポーネントの統合が機能していることを確認できる
- このテストは `generate-config` コマンドも同時に検証する
- 手動で YAML を作成する必要が一切ない
- 生成された設定が実際に動作することを保証

---

## 10. Test Data & Fixtures (Using pipeline-app)

**Strategy**: 既存の `apps/pipeline-app` をテストデータとして活用。Mockデータは一切作成しない。

### 10.1 Test Skeleton Definition

**Location**: `packages/xform-core/tests/dag/fixtures/test_skeleton.py`

```python
"""Test skeleton using pipeline-app transforms."""

from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep
from algo_trade_dtypes.generators import HLOCVSpec
from algo_trade_dtypes.types import FeatureMap
import pandas as pd

# Test skeleton with pipeline-app transforms
test_pipeline_skeleton = PipelineSkeleton(
    name="test_pipeline",
    steps=[
        PipelineStep(
            name="generate_bars",
            input_types=(HLOCVSpec,),
            output_type=pd.DataFrame,
            default_transform="pipeline_app.transforms.generate_price_bars",
        ),
        PipelineStep(
            name="compute_features",
            input_types=(pd.DataFrame,),
            output_type=FeatureMap,
            default_transform="pipeline_app.transforms.compute_feature_map",
        ),
        PipelineStep(
            name="select_features",
            input_types=(FeatureMap,),
            output_type=list[str],
            default_transform="pipeline_app.transforms.select_top_features",
        ),
    ],
)
```

### 10.2 Available Transforms (from pipeline-app)

**実際の Transform 関数**:

| Transform FQN | Input Types | Output Type | Parameters |
|--------------|-------------|-------------|------------|
| `pipeline_app.transforms.generate_price_bars` | (HLOCVSpec,) | pd.DataFrame | - |
| `pipeline_app.transforms.compute_feature_map` | (pd.DataFrame,) | FeatureMap | annualization_factor: float = 252.0 |
| `pipeline_app.transforms.select_top_features` | (FeatureMap,) | list[str] | top_n: int = 2, minimum_score: float = 0.0 |
| `pipeline_app.transforms.merge_feature_maps` | (FeatureMap, FeatureMap) | FeatureMap | prefix_a: str = "a_", prefix_b: str = "b_" |
| `pipeline_app.transforms.compute_weighted_score` | (FeatureMap, FeatureMap) | float | normalize: bool = True |
| `pipeline_app.transforms.split_features_by_threshold` | (FeatureMap,) | tuple[FeatureMap, FeatureMap] | threshold: float = 0.0 |

### 10.3 Configuration Generation Strategy (No Manual YAML)

**Approach**: 手動でYAML設定ファイルを作成せず、`generate-config` CLIコマンドで自動生成。

```python
"""Test fixture for config generation."""

import pytest
from pathlib import Path
import subprocess
import yaml

@pytest.fixture
def generated_config_dir(tmp_path):
    """Directory for generated config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def valid_config_path(generated_config_dir, test_skeleton_registered):
    """Generate valid config using CLI command."""
    output_path = generated_config_dir / "valid_config.yaml"
    
    # Call generate-config CLI command
    result = subprocess.run([
        "uv", "run", "python", "-m", "xform_core.dag",
        "apps/pipeline-app",
        "generate-config",
        "--skeleton", "test_pipeline_skeleton",
        "--output", str(output_path),
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Config generation failed: {result.stderr}"
    assert output_path.exists(), "Config file not created"
    
    return output_path

@pytest.fixture
def invalid_config_missing_step(generated_config_dir):
    """Create invalid config with missing required step."""
    config_path = generated_config_dir / "invalid_missing_step.yaml"
    config = {
        "pipeline": {
            "name": "test-pipeline",
            "skeleton": "test_pipeline_skeleton"
        },
        "test_pipeline": {
            "steps": {}  # Missing required steps
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def invalid_config_unknown_param(valid_config_path, generated_config_dir):
    """Modify generated config to add unknown parameter."""
    # Load valid config
    with open(valid_config_path) as f:
        config = yaml.safe_load(f)
    
    # Add unknown parameter
    if "compute_features" in config.get("test_pipeline", {}).get("steps", {}):
        config["test_pipeline"]["steps"]["compute_features"]["params"]["unknown_param"] = "invalid"
    
    # Save as invalid config
    invalid_path = generated_config_dir / "invalid_unknown_param.yaml"
    with open(invalid_path, "w") as f:
        yaml.dump(config, f)
    
    return invalid_path

# Similarly for other invalid configs...
```

**Benefits**:
- ✅ No manual YAML maintenance
- ✅ Tests `generate-config` command functionality
- ✅ Ensures generated configs are actually valid
- ✅ Invalid configs are programmatically created from valid ones
- ✅ Full CLI workflow tested (generate → validate → run)
- ✅ Configs always in sync with skeleton definitions

### 10.4 Test Input Data

```python
"""Test fixtures using pipeline-app types."""

from algo_trade_dtypes.generators import HLOCVSpec
from algo_trade_dtypes.types import FeatureMap

# Sample input for testing
sample_hlocv_spec = HLOCVSpec(
    symbol="TEST",
    start_date="2024-01-01",
    end_date="2024-01-31",
    frequency="1D",
    n_bars=30,
)

sample_feature_map: FeatureMap = {
    "mean_return": 0.05,
    "volatility": 0.15,
    "sharpe_ratio": 0.33,
    "drawdown": 0.02,
}

sample_weights: FeatureMap = {
    "mean_return": 0.4,
    "volatility": 0.3,
    "sharpe_ratio": 0.2,
    "drawdown": 0.1,
}
```

### 10.5 Test Registry Setup

```python
"""Fixture for populating test registry with pipeline-app transforms."""

import pytest
from xform_core.transform_registry import TransformRegistry, TransformSignature
from xform_core.dag.skeleton import register_skeleton
import importlib

@pytest.fixture
def test_registry() -> TransformRegistry:
    """Create registry populated with pipeline-app transforms."""
    registry = TransformRegistry()
    
    # Import pipeline_app.transforms to trigger @transform registration
    import pipeline_app.transforms
    
    # Transforms are auto-registered via @transform decorator
    # Just return the global registry
    from xform_core.transform_registry import get_registry
    return get_registry()

@pytest.fixture
def test_skeleton_registered():
    """Register test skeleton."""
    from tests.dag.fixtures.test_skeleton import test_pipeline_skeleton
    register_skeleton("test_pipeline_skeleton", test_pipeline_skeleton)
    yield
    # Cleanup after test if needed
```

### 10.6 Benefits of Using pipeline-app

✅ **No Mock Creation**: 既存の実装を活用、Mock不要  
✅ **Real Transforms**: 実際の `@transform` 関数でテスト  
✅ **Real Types**: `FeatureMap`, `HLOCVSpec` など実際の型  
✅ **Real Checks**: `Check` 関数も実際のものを使用  
✅ **Variety**: 6つの Transform で多様なシグネチャをカバー  
✅ **Maintainability**: pipeline-app の更新に自動的に追従

---

## 11. Test Coverage Goals (Revised for Minimal Testing)

### 11.1 Component Coverage

| Component | Target Coverage | Critical Paths | Test Count |
|-----------|----------------|----------------|------------|
| Transform Registry | 80%+ | Type matching, signature validation | 3 tests |
| Skeleton | 75%+ | Registration, config validation | 2 tests |
| **Configuration Validator** | **95%+** | **All error types, suggestions** | **17 tests** |
| Transform Resolver | 75%+ | Resolution logic, default handling | 3 tests |
| DAG Executor | 80%+ | Validation integration, Fail Fast | 3 tests |
| CLI Commands | 70%+ | Basic commands, error handling | 5 tests |
| Unified CLI | 70%+ | App detection, skeleton import | 2 tests |
| Configuration Loading | 75%+ | Load, auto-validate, Fail Fast | 3 tests |
| Integration | 80%+ | E2E workflow | 1 test |

**Total Test Count**: 約 **39 tests** (前回の100ケースから大幅削減)

### 11.2 Error Path Coverage (Configuration Validator のみ詳細)

**All error types in ConfigurationValidator must have dedicated tests:**
- ✅ MISSING_REQUIRED_STEP (CV-E-01)
- ✅ NO_TRANSFORM_SPECIFIED (CV-E-02)
- ✅ TRANSFORM_NOT_FOUND (CV-E-03)
- ✅ TYPE_SIGNATURE_MISMATCH (CV-E-04, CV-E-05)
- ✅ UNKNOWN_PARAMETER (CV-E-06)
- ✅ MISSING_REQUIRED_PARAMETER (CV-E-07)
- ✅ PARAMETER_TYPE_MISMATCH (CV-E-08)
- ✅ UNKNOWN_STEP (CV-W-01, warning)

---

## 12. Test Execution Strategy

### 12.1 Test Organization

```
packages/xform-core/tests/dag/
├── __init__.py
├── conftest.py                          # Shared fixtures (test_registry, config generation, etc.)
├── fixtures/
│   ├── __init__.py
│   └── test_skeleton.py                 # Test skeleton using pipeline-app
├── test_transform_registry_enhanced.py  # TR-* tests
├── test_dag_skeleton.py                 # SK-* tests
├── test_configuration_validator.py      # CV-* tests (CRITICAL)
├── test_transform_resolver.py           # RS-* tests
├── test_dag_executor.py                 # EX-* tests
├── test_dag_cli.py                      # CLI-* tests
├── test_dag_main.py                     # MAIN-* tests
├── test_pipeline_config.py              # CFG-* tests
└── test_dag_integration.py              # INT-* tests (with generate-config workflow)
```

**Note**: 
- ❌ `fixtures/test_configs/` ディレクトリは不要（手動YAMLファイル不要）
- ✅ 設定ファイルは全て `generate-config` CLI コマンドで自動生成
- ✅ Invalid configs は valid config を programmatically 改変して作成

### 12.2 conftest.py Setup

```python
"""Shared fixtures for DAG tests using pipeline-app."""

import pytest
import subprocess
import yaml
from pathlib import Path
from xform_core.transform_registry import get_registry
from xform_core.dag.skeleton import register_skeleton

@pytest.fixture(scope="session", autouse=True)
def import_pipeline_app():
    """Import pipeline-app transforms to populate registry."""
    import pipeline_app.transforms
    # Transforms auto-registered via @transform decorator
    yield

@pytest.fixture(scope="session")
def test_registry():
    """Get global registry populated with pipeline-app transforms."""
    return get_registry()

@pytest.fixture
def test_skeleton_registered():
    """Register test skeleton for tests."""
    from tests.dag.fixtures.test_skeleton import test_pipeline_skeleton
    register_skeleton("test_pipeline_skeleton", test_pipeline_skeleton)
    yield
    # Cleanup if needed

@pytest.fixture
def generated_config_dir(tmp_path):
    """Directory for generated config files (per-test temporary)."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def valid_config_path(generated_config_dir, test_skeleton_registered):
    """Generate valid config using generate-config CLI command."""
    output_path = generated_config_dir / "valid_config.yaml"
    
    result = subprocess.run([
        "uv", "run", "python", "-m", "xform_core.dag",
        "apps/pipeline-app",
        "generate-config",
        "--skeleton", "test_pipeline_skeleton",
        "--output", str(output_path),
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Config generation failed: {result.stderr}"
    assert output_path.exists(), "Config file not created"
    
    return output_path

@pytest.fixture
def invalid_config_missing_step(generated_config_dir):
    """Create invalid config with missing required step."""
    config_path = generated_config_dir / "invalid_missing_step.yaml"
    config = {
        "pipeline": {
            "name": "test-pipeline",
            "skeleton": "test_pipeline_skeleton"
        },
        "test_pipeline": {
            "steps": {}  # Missing required steps
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def invalid_config_unknown_param(valid_config_path, generated_config_dir):
    """Modify generated config to add unknown parameter."""
    with open(valid_config_path) as f:
        config = yaml.safe_load(f)
    
    # Add unknown parameter to first step with params
    if "test_pipeline" in config and "steps" in config["test_pipeline"]:
        for step_name, step_config in config["test_pipeline"]["steps"].items():
            if "params" in step_config:
                step_config["params"]["unknown_param"] = "invalid"
                break
    
    invalid_path = generated_config_dir / "invalid_unknown_param.yaml"
    with open(invalid_path, "w") as f:
        yaml.dump(config, f)
    
    return invalid_path

@pytest.fixture
def sample_hlocv_spec():
    """Sample HLOCVSpec for testing."""
    from algo_trade_dtypes.generators import HLOCVSpec
    return HLOCVSpec(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-01-31",
        frequency="1D",
        n_bars=30,
    )

@pytest.fixture
def sample_feature_map():
    """Sample FeatureMap for testing."""
    from algo_trade_dtypes.types import FeatureMap
    return FeatureMap(
        mean_return=0.05,
        volatility=0.15,
        sharpe_ratio=0.33,
        drawdown=0.02,
    )
```

**Key Changes**:
- ❌ Removed `config_fixtures_dir` (no static YAML files)
- ✅ Added `generated_config_dir` (temporary directory per test)
- ✅ Added `valid_config_path` (generates config via CLI)
- ✅ Added `invalid_config_*` fixtures (programmatically created)
- ✅ All configs generated dynamically, no maintenance burden

### 12.3 Test Execution Order

1. **Setup** (自動):
   - `import_pipeline_app` fixture で pipeline-app transforms を import
   - Transform Registry に自動登録
   - Test skeleton を登録

2. **Unit Tests** (並列実行可能):
   - Transform Registry (TR-*) - 3 tests
   - Skeleton (SK-*) - 2 tests
   - Resolver (RS-*) - 3 tests
   
3. **Core Logic Tests** (順次実行推奨):
   - Configuration Validator (CV-*) - **17 tests (最重要)**
   - DAG Executor (EX-*) - 3 tests
   - Configuration Loading (CFG-*) - 3 tests

4. **Interface Tests**:
   - CLI Commands (CLI-*) - 5 tests
   - Unified CLI (MAIN-*) - 2 tests

5. **Integration Tests** (最後):
   - End-to-End (INT-*) - 1 test

### 12.4 Test Commands

```bash
# IMPORTANT: Tests depend on pipeline-app being in the workspace
# Ensure apps/pipeline-app is present

# Run all DAG tests with pipeline-app
uv run pytest packages/xform-core/tests/dag/ -v

# Run specific component tests
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py -v

# Run with coverage report
uv run pytest packages/xform-core/tests/dag/ \
  --cov=xform_core.dag \
  --cov-report=term-missing \
  --cov-report=html:output/coverage/dag

# Run only critical validator tests (17 tests)
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py -v -k "CV"

# Run with pipeline-app integration check
uv run pytest packages/xform-core/tests/dag/ -v --tb=short

# Debug single test
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py::test_cv_e01_missing_required_step -vv
```

### 12.5 Test Dependencies

**Required Packages** (from workspace):
- `xform-core` - Core DAG infrastructure
- `pipeline-app` - Transform implementations and types
- `algo-trade-dtypes` - Type definitions (`FeatureMap`, `HLOCVSpec`)
- `pytest` - Test framework

**Test Isolation**:
- Each test is independent (no shared state)
- Registry is populated once per session (via `import_pipeline_app` fixture)
- Config files are read-only fixtures
- Cleanup handled by pytest fixtures

---

## 13. Test Success Criteria (Revised)

### 13.1 Must Pass (Blocking) - 23 tests

**Configuration Validator (17 tests)**:
- ✅ All CV-N-* tests pass (4 tests: valid configs)
- ✅ All CV-E-* tests pass (8 tests: error detection)
- ✅ All CV-W-* tests pass (1 test: warnings)
- ✅ All CV-M-* tests pass (2 tests: multiple errors)
- ✅ All CV-S-* tests pass (3 tests: suggestion quality)

**Fail Fast Validation (3 tests)**:
- ✅ EX-E-01 passes (Executor validation prevents execution)
- ✅ CFG-E-01 passes (Config load validation)
- ✅ CLI-E-02 passes (CLI validation before run)

**E2E Integration (1 test)**:
- ✅ INT-E2E-01 passes (Full workflow)

**Other Components Happy Path (5 tests)**:
- ✅ TR-N-01, SK-N-01, RS-N-01, EX-N-01, CLI-N-01 pass

### 13.2 Should Pass (High Priority) - 16 tests

- ✅ All remaining *-N-* tests (正常系)
- ✅ All remaining *-E-* tests (主要な異常系)

**Total Critical Tests**: 39 tests (Must + Should)

---

## 14. Test Maintenance Guidelines

### 14.1 Adding New Error Types

When adding new validation error types:
1. Add test case in CV-E-* series
2. Verify suggestion quality (CV-S-* series)
3. Update error type coverage checklist (Section 11.2)

### 14.2 Adding New Components

When adding new components:
1. Create dedicated test file
2. Follow naming convention: test_<component>.py
3. Include N-01, E-01, B-01 tests minimum
4. Update test coverage goals (Section 11.1)

### 14.3 Regression Testing

All bug fixes should include:
1. New test case reproducing the bug
2. Verification that fix resolves the test
3. Add to regression test suite

---

## Summary

このテスト仕様は以下を保証する：

### 設計方針
1. **最小限の識別**: 各コンポーネントは統合型正常系1ケース + 代表的な異常系1-2ケース
2. **Configuration Validator に集中**: 全エラータイプ（8種類）を個別に詳細テスト
3. **Fail Fast の徹底検証**: 実行前バリデーションが機能することを複数箇所で確認
4. **型安全性**: 型シグネチャ検証ロジックを重点的にテスト（CV-E-04, CV-E-05, TR-E-02）
5. **実用性**: CLI インターフェース、設定生成、E2E ワークフローをカバー

### テスト統計

| カテゴリ | テスト数 | 備考 |
|---------|---------|------|
| Transform Registry | 3 | 統合型正常系 + 主要異常系 |
| Skeleton | 2 | 統合型正常系 + 異常系 |
| **Configuration Validator** | **17** | **全エラータイプ詳細カバー** |
| Transform Resolver | 3 | 正常系2 + 異常系1 |
| DAG Executor | 3 | 統合型正常系 + Fail Fast検証 |
| CLI Commands | 5 | 基本コマンド + エラー処理 |
| Unified CLI | 2 | 統合型正常系 + 異常系 |
| Configuration Loading | 3 | 統合型正常系 + Fail Fast |
| Integration | 1 | E2E ワークフロー |
| **合計** | **39** | **前回100ケースから60%削減** |

### 重点テスト領域

**Configuration Validator (17 tests / 44% of total)**:
- 正常系: 4 tests (基本、デフォルト、オプション、型注釈)
- 異常系: 8 tests (全エラータイプ個別)
- 警告系: 1 test
- 複合系: 2 tests (複数エラー、エラー+警告)
- Suggestion品質: 3 tests

**Fail Fast Validation (3 tests / 8% of total)**:
- Executor validation (EX-E-01)
- Config load validation (CFG-E-01)
- CLI validation (CLI-E-02)

**他のコンポーネント (19 tests / 49% of total)**:
- 各コンポーネント最小限の正常系・異常系
- E2E 統合テスト

### 削減の根拠

Configuration Validator 以外のコンポーネントは：
- ✅ 複数メソッドを1つの統合型テストで検証
- ✅ 異常系は代表的なエラーパターンのみ
- ✅ 境界値テストは省略（必要なら実装時に追加）
- ✅ 詳細なバリエーションは Configuration Validator で担保

### pipeline-app 活用戦略

**Mock不要の実装**:
- ✅ 既存の `apps/pipeline-app` を完全活用
- ✅ 6つの実際の Transform 関数（多様なシグネチャ）
- ✅ 実際の型定義 (`FeatureMap`, `HLOCVSpec`, `pd.DataFrame`)
- ✅ 実際の Check 関数
- ✅ テストコードのメンテナンス性向上
- ✅ pipeline-app の更新に自動追従

**Config Generation Strategy**:
- ✅ 手動 YAML ファイル不要
- ✅ `generate-config` CLI で自動生成
- ✅ Invalid configs は valid config を改変して作成
- ✅ Full CLI workflow をテスト (generate → validate → run)
- ✅ Skeleton と config の同期を自動保証

**Transform バリエーション** (from pipeline-app):
1. Single input, no params: `generate_price_bars(HLOCVSpec) -> pd.DataFrame`
2. Single input, optional params: `compute_feature_map(pd.DataFrame, *, annualization_factor=252.0) -> FeatureMap`
3. Single input, multiple params: `select_top_features(FeatureMap, *, top_n=2, minimum_score=0.0) -> list[str]`
4. Multiple inputs, params: `merge_feature_maps(FeatureMap, FeatureMap, *, prefix_a="a_", prefix_b="b_") -> FeatureMap`
5. Multiple outputs (tuple): `split_features_by_threshold(FeatureMap, threshold=0.0) -> tuple[FeatureMap, FeatureMap]`
6. Complex scoring: `compute_weighted_score(FeatureMap, FeatureMap, *, normalize=True) -> float`

→ **これらで型シグネチャのほぼ全パターンをカバー可能**

**最重要テスト**: Configuration Validator (CV-*) - 全エラータイプの検出と Suggestion 品質

---

## Quick Reference: Implementation Checklist

### Phase 1: Setup Test Infrastructure

- [ ] Create `packages/xform-core/tests/dag/` directory structure
- [ ] Create `fixtures/` subdirectory with `test_skeleton.py`
- [ ] ~~Create `fixtures/test_configs/` with sample YAML files~~ ❌ **Not needed** (use `generate-config` CLI)
- [ ] Implement `conftest.py` with:
  - pipeline-app integration
  - `generated_config_dir` fixture (tmp_path based)
  - `valid_config_path` fixture (calls `generate-config` CLI)
  - `invalid_config_*` fixtures (programmatic creation)

### Phase 2: Implement Core Component Tests (22 tests)

- [ ] Transform Registry (3 tests) - TR-N-01, TR-E-01, TR-E-02
- [ ] Skeleton (2 tests) - SK-N-01, SK-E-01
- [ ] Transform Resolver (3 tests) - RS-N-01, RS-N-02, RS-E-01
- [ ] DAG Executor (3 tests) - EX-N-01, EX-E-01, EX-E-02
- [ ] Configuration Loading (3 tests) - CFG-N-01, CFG-E-01, CFG-E-02
- [ ] CLI Commands (5 tests) - CLI-N-01~03, CLI-E-01~02
- [ ] Unified CLI (2 tests) - MAIN-N-01, MAIN-E-01
- [ ] Integration (1 test) - INT-E2E-01

### Phase 3: Implement Configuration Validator Tests (17 tests) - **CRITICAL**

- [ ] **正常系 (4 tests)**: CV-N-01~04
- [ ] **異常系 (8 tests)**: CV-E-01~08 (各エラータイプ個別)
- [ ] **警告系 (1 test)**: CV-W-01
- [ ] **複合系 (2 tests)**: CV-M-01~02
- [ ] **Suggestion品質 (3 tests)**: CV-S-01~03

### Verification

```bash
# Run all tests
uv run pytest packages/xform-core/tests/dag/ -v

# Verify test count: should be 39
uv run pytest packages/xform-core/tests/dag/ --collect-only | grep "test session starts" -A 1

# Verify Configuration Validator coverage: should be 17
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py --collect-only

# Run coverage check
uv run pytest packages/xform-core/tests/dag/ --cov=xform_core.dag --cov-report=term-missing
```

### Success Criteria

✅ All 39 tests pass  
✅ Configuration Validator: 17 tests, all error types covered  
✅ Coverage: 80%+ for all components  
✅ No Mock code (all using pipeline-app)  
✅ No manual YAML files (all using `generate-config` CLI)  
✅ E2E integration test passes:
  - `generate-config` successfully creates valid YAML
  - `validate` command verifies generated config
  - `run` command executes pipeline with generated config  

### E2E Workflow Verification

```bash
# Manual verification of the full workflow
cd /Users/mikke/git_dir/TransformFn

# 1. Generate config from test skeleton
uv run python -m xform_core.dag apps/pipeline-app generate-config \
  --skeleton test_pipeline_skeleton \
  --output /tmp/test_generated.yaml

# 2. Validate generated config
uv run python -m xform_core.dag apps/pipeline-app validate /tmp/test_generated.yaml

# 3. Run pipeline with generated config
uv run python -m xform_core.dag apps/pipeline-app run /tmp/test_generated.yaml

# Expected output:
# ✓ Config generated
# ✓ Validation passed
# ✓ Pipeline completed successfully
```

---

## Related Documentation

- **Implementation Spec**: [`doc/transformfn_spec/dag-dynamic-transform-selection.md`](../doc/transformfn_spec/dag-dynamic-transform-selection.md) - 実装設計の詳細
- **Migration Guide**: [`doc/MIGRATION_TO_DYNAMIC_DAG.md`](../doc/MIGRATION_TO_DYNAMIC_DAG.md) - 手続き型DAGからの移行ガイド
- **Architecture**: [`doc/ARCHITECTURE.md`](../doc/ARCHITECTURE.md) - 全体アーキテクチャ
- **Pipeline App**: [`apps/pipeline-app/`](../apps/pipeline-app/) - テストで使用するサンプルアプリ
- **Transform Guidelines**: [`CLAUDE.md`](../CLAUDE.md) - Transform関数ガイドライン
