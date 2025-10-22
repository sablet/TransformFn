# DAG Dynamic Transform Selection Design

## 概要

パイプラインの**型変換構造（Skeleton）**と**具体的なTransform実装**を分離し、設定ファイルで動的に関数を選択できるようにする設計。

## 設計原則

### 1. Type-Driven Pipeline Skeleton

```python
# パイプライン構造は型シグネチャで定義
class Phase1Skeleton:
    step1: (MarketDataIngestionConfig,) -> ProviderBatchCollection
    step2: (ProviderBatchCollection,) -> NormalizedOHLCVBundle
    step3: (NormalizedOHLCVBundle,) -> MultiAssetOHLCVFrame
    step4: (MultiAssetOHLCVFrame, MarketDataIngestionConfig) -> MarketDataSnapshotMeta
```

### 2. Transform Selection by Configuration

```yaml
phase1:
  steps:
    step1:
      transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"
      params:
        use_adjusted_close: true
    step2:
      transform: "algo_trade_transforms.market_data.normalize_multi_provider"
      params:
        target_frequency: "1H"
```

### 3. Type-Based Transform Discovery

Auditor が収集した Transform 関数を型シグネチャでフィルタリングし、利用可能な選択肢を提示:

```python
# 型シグネチャが一致する関数を検索
candidates = registry.find_transforms(
    input_types=(MarketDataIngestionConfig,),
    output_type=ProviderBatchCollection
)
# -> ["fetch_yahoo_finance_ohlcv", "fetch_ccxt_ohlcv", ...]
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│ DAG Configuration (YAML)                                    │
│ [App-specific: apps/algo-trade/configs/]                   │
│ - Pipeline structure reference                              │
│ - Transform function selection (FQN)                        │
│ - Parameters for each transform                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Configuration Validator (CRITICAL)                          │
│ [Core: xform_core.dag.validator]                           │
│ ✓ Transform FQN existence check                            │
│ ✓ Type signature compatibility validation                   │
│ ✓ Parameter schema validation                               │
│ ✓ Required parameter completeness check                     │
│ ✓ Skeleton step coverage validation                         │
│ → FAIL FAST before execution                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ DAG Skeleton (Python)                                       │
│ [Core base classes: xform_core.dag.skeleton]               │
│ [App-specific definitions: algo_trade_dag.skeleton]        │
│ - Type transformation flow definition                       │
│ - Step dependencies                                         │
│ - Utility function integration points                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Transform Resolver                                          │
│ [Core: xform_core.dag.resolver]                            │
│ - Resolve FQN to actual function                           │
│ - Validate type signature compatibility                     │
│ - Inject parameters                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Transform Registry (Enhanced)                               │
│ [Core: xform_core.transform_registry]                      │
│ - Index by input/output type signature                     │
│ - Store function metadata (FQN, types, params)             │
│ - Query available transforms for type conversion           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ DAG Executor                                                │
│ [Core: xform_core.dag.executor]                            │
│ - Execute resolved transforms                               │
│ - Handle caching                                            │
│ - Collect execution results                                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Distribution

**Core Components** (`packages/xform-core/`):
- 再利用可能なDAG実行基盤（550行程度）
- 型駆動のバリデーション・解決・実行ロジック
- **統一CLIエントリーポイント（`python -m xform_core.dag`）**
- 全アプリケーションで共通利用

**App-Specific Components** (`apps/algo-trade/`):
- 具体的なパイプライン構造（Skeleton）定義（100行程度）
- アプリ固有のTransform実装（既存）
- アプリ固有の型定義（既存）

**削減効果**: 新しいアプリを作る場合、**約100行のSkeleton定義だけで完全なDAGパイプラインCLIが完成**（CLIコード不要）

## 実装コンポーネント配置

### Core Components (`packages/xform-core/xform_core/dag/`)

汎用的なDAG実行基盤（全アプリで再利用可能）:
- `skeleton.py` - Skeleton定義の基底クラスとユーティリティ
- `validator.py` - Configuration Validator（型・パラメータバリデーション）
- `resolver.py` - Transform Resolver（FQN→関数解決）
- `executor.py` - DAG Executor（バリデーション統合済み実行エンジン）
- `config.py` - Configuration loading utilities
- `cli.py` - CLI command implementations（validate/run/discover）
- **`__main__.py` - 統一CLIエントリーポイント（アプリ自動検出・Skeleton動的ロード）**

### App-Specific Components (`apps/algo-trade/algo_trade_dag/`)

アプリ固有のパイプライン定義:
- `skeleton.py` - Phase1-4の具体的なSkeleton定義（型変換フロー） **[必須]**

**それだけです！** CLIエントリーポイントは不要。Core側の `__main__.py` がアプリを自動検出します。

---

## 実装コンポーネント詳細

### 1. Enhanced Transform Registry (Core)

**Location**: `packages/xform-core/xform_core/transform_registry.py`

```python
from typing import Callable, Type, Any
from dataclasses import dataclass

@dataclass
class TransformSignature:
    """Type signature of a transform function."""
    input_types: tuple[Type[Any], ...]
    output_type: Type[Any]
    params: dict[str, Any]

class TransformRegistry:
    """Registry for transform functions with type-based indexing."""
    
    def register(
        self,
        fqn: str,
        func: Callable,
        signature: TransformSignature,
    ) -> None:
        """Register a transform with its type signature."""
        ...
    
    def find_transforms(
        self,
        input_types: tuple[Type[Any], ...],
        output_type: Type[Any],
    ) -> list[str]:
        """Find transforms matching the type signature."""
        ...
    
    def get_transform(self, fqn: str) -> Callable:
        """Resolve FQN to actual function."""
        ...
    
    def validate_signature(
        self,
        fqn: str,
        input_types: tuple[Type[Any], ...],
        output_type: Type[Any],
    ) -> bool:
        """Validate that transform signature matches expected types."""
        ...
```

### 2. DAG Skeleton Definition (Core Base Classes)

**Location**: `packages/xform-core/xform_core/dag/skeleton.py`

```python
from dataclasses import dataclass
from typing import Type, Any

@dataclass
class PipelineStep:
    """Single step in pipeline skeleton."""
    name: str
    input_types: tuple[Type[Any], ...]
    output_type: Type[Any]
    default_transform: str | None = None
    required: bool = True

@dataclass
class PipelineSkeleton:
    """Pipeline structure definition (reusable across apps)."""
    name: str
    steps: list[PipelineStep]
    
    def validate_config(self, config: dict) -> bool:
        """Validate that config provides all required steps."""
        required_steps = {step.name for step in self.steps if step.required}
        config_steps = set(config.get("steps", {}).keys())
        
        missing_steps = required_steps - config_steps
        for step_name in missing_steps:
            step = next(s for s in self.steps if s.name == step_name)
            if step.default_transform is None:
                return False
        
        return True


# Skeleton registry for dynamic lookup
_SKELETON_REGISTRY: dict[str, PipelineSkeleton] = {}

def register_skeleton(fqn: str, skeleton: PipelineSkeleton) -> None:
    """Register a skeleton for dynamic lookup."""
    _SKELETON_REGISTRY[fqn] = skeleton

def get_skeleton(fqn: str) -> PipelineSkeleton:
    """Get skeleton by fully qualified name."""
    if fqn not in _SKELETON_REGISTRY:
        raise ValueError(f"Skeleton '{fqn}' not found in registry")
    return _SKELETON_REGISTRY[fqn]
```

#### App-Specific Skeleton Definition

**Location**: `apps/algo-trade/algo_trade_dag/skeleton.py`

```python
"""Algo-trade specific pipeline skeleton definitions."""

from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep, register_skeleton
from algo_trade_dtypes.types import (
    MarketDataIngestionConfig,
    ProviderBatchCollection,
    NormalizedOHLCVBundle,
    MultiAssetOHLCVFrame,
    MarketDataSnapshotMeta,
)

# Phase 1 Skeleton: Market Data Ingestion
phase1_skeleton = PipelineSkeleton(
    name="phase1_market_data_ingestion",
    steps=[
        PipelineStep(
            name="fetch_data",
            input_types=(MarketDataIngestionConfig,),
            output_type=ProviderBatchCollection,
            default_transform="algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv",
        ),
        PipelineStep(
            name="normalize",
            input_types=(ProviderBatchCollection,),
            output_type=NormalizedOHLCVBundle,
            default_transform="algo_trade_transforms.market_data.normalize_multi_provider",
        ),
        PipelineStep(
            name="merge",
            input_types=(NormalizedOHLCVBundle,),
            output_type=MultiAssetOHLCVFrame,
            default_transform="algo_trade_transforms.market_data.merge_market_data_bundle",
        ),
        PipelineStep(
            name="persist",
            input_types=(MultiAssetOHLCVFrame, MarketDataIngestionConfig),
            output_type=MarketDataSnapshotMeta,
            default_transform="algo_trade_transforms.market_data.persist_market_data_snapshot",
        ),
    ],
)

# Register for dynamic lookup
register_skeleton("algo_trade_dag.skeleton.phase1_skeleton", phase1_skeleton)

# Similarly for phase2, phase3, phase4...
# (省略: phase2_skeleton, phase3_skeleton, phase4_skeleton)
```

### 6. Unified CLI Entry Point (Core)

#### CLI Command Implementations

**Location**: `packages/xform-core/xform_core/dag/cli.py`

```python
"""CLI command implementations (used by __main__.py)."""

from typing import Any
from xform_core.dag.config import PipelineConfig
from xform_core.dag.executor import DAGExecutor
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.validator import ConfigurationValidator
from xform_core.dag.skeleton import get_skeleton
from xform_core.transform_registry import get_registry


def validate_command(config_path: str) -> int:
    """Validate configuration without execution."""
    try:
        config = PipelineConfig(config_path)
        print(f"✓ Configuration {config_path} is valid")
        return 0
    except ValueError as e:
        print(f"✗ Configuration validation failed:\n{e}")
        return 1


def run_command(config_path: str, initial_inputs: dict[str, Any] | None = None) -> int:
    """Run pipeline from configuration."""
    try:
        config = PipelineConfig(config_path)
        
        skeleton_fqn = config.raw["pipeline"]["skeleton"]
        skeleton = get_skeleton(skeleton_fqn)
        
        registry = get_registry()
        resolver = TransformResolver(registry)
        validator = ConfigurationValidator(registry, skeleton)
        executor = DAGExecutor(skeleton, resolver, validator)
        
        print(f"Running pipeline: {skeleton.name}")
        result = executor.execute(config.raw, initial_inputs or {})
        
        print(f"✓ Pipeline completed successfully")
        print(f"Results: {result}")
        return 0
        
    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def discover_command(input_type: str | None = None, output_type: str | None = None) -> int:
    """Discover available transforms."""
    registry = get_registry()
    
    if input_type and output_type:
        print(f"Searching transforms: {input_type} -> {output_type}")
        # TODO: Type resolution and discovery
        print("(Not implemented yet)")
    else:
        print("All registered transforms:")
        # TODO: List all transforms
        print("(Not implemented yet)")
    
    return 0


def generate_config_command(
    skeleton: str | None = None,
    generate_all: bool = False,
    output: str | None = None,
    output_dir: str | None = None,
    show_alternatives: bool = False,
) -> int:
    """Generate sample configuration from skeleton.
    
    Process:
    1. Get skeleton by name from registry
    2. For each step:
        - Select transform: default_transform or first candidate from registry
        - Extract parameters from function signature
        - Get default values for parameters
    3. Generate YAML with:
        - Pipeline metadata (name, skeleton reference)
        - Steps with transforms and parameters
        - Comments with alternatives (if show_alternatives)
    """
    from xform_core.dag.skeleton import get_skeleton, _SKELETON_REGISTRY
    from xform_core.transform_registry import get_registry
    
    registry = get_registry()
    
    if generate_all:
        # Generate for all registered skeletons
        for skeleton_fqn in _SKELETON_REGISTRY.keys():
            skeleton_obj = get_skeleton(skeleton_fqn)
            output_path = f"{output_dir}/{skeleton_obj.name}.yaml"
            _generate_single_config(skeleton_obj, registry, output_path, show_alternatives)
        print(f"✓ Generated {len(_SKELETON_REGISTRY)} config files in {output_dir}")
        return 0
    
    if not skeleton:
        print("✗ Error: --skeleton or --all required")
        return 1
    
    # Generate single skeleton
    skeleton_obj = get_skeleton(skeleton)
    _generate_single_config(skeleton_obj, registry, output, show_alternatives)
    print(f"✓ Generated config: {output}")
    return 0
```

#### Unified CLI Entry Point

**Location**: `packages/xform-core/xform_core/dag/__main__.py`

xform_auditor と同じパターンで、アプリを動的に検出・ロード:

```python
"""Unified DAG CLI entry point - handles all apps dynamically.

Usage:
    uv run python -m xform_core.dag apps/algo-trade validate configs/pipeline.yaml
    uv run python -m xform_core.dag apps/algo-trade run configs/pipeline.yaml
    uv run python -m xform_core.dag apps/algo-trade discover
"""

import sys
import argparse
from pathlib import Path
from importlib import import_module

from xform_core.dag.cli import validate_command, run_command, discover_command


def main() -> int:
    """Main CLI entry point with automatic app detection."""
    parser = argparse.ArgumentParser(
        description="DAG pipeline execution (unified across all apps)"
    )
    parser.add_argument(
        "app_path",
        help="App directory path (e.g., apps/algo-trade)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate pipeline configuration",
    )
    validate_parser.add_argument("config", help="Path to config YAML")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline from configuration")
    run_parser.add_argument("config", help="Path to config YAML")
    
    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover available transforms",
    )
    discover_parser.add_argument("--input-type", help="Input type to search for")
    discover_parser.add_argument("--output-type", help="Output type to search for")
    
    # Generate-config command
    generate_parser = subparsers.add_parser(
        "generate-config",
        help="Generate sample config from skeleton",
    )
    generate_parser.add_argument("--skeleton", help="Skeleton name (e.g., phase1_skeleton)")
    generate_parser.add_argument("--all", action="store_true", help="Generate for all skeletons")
    generate_parser.add_argument("--output", help="Output file path")
    generate_parser.add_argument("--output-dir", help="Output directory (for --all)")
    generate_parser.add_argument("--show-alternatives", action="store_true", 
                                  help="Include alternative transforms as comments")
    
    args = parser.parse_args()
    
    # Resolve and import app skeleton module to register skeletons
    app_module = _resolve_app_module(args.app_path)
    try:
        import_module(f"{app_module}.skeleton")
    except ImportError as e:
        print(f"✗ Failed to import {app_module}.skeleton: {e}")
        print(f"  Ensure {args.app_path}/{app_module}/skeleton.py exists")
        return 1
    
    # Execute command
    if args.command == "validate":
        return validate_command(args.config)
    elif args.command == "run":
        return run_command(args.config)
    elif args.command == "discover":
        return discover_command(
            getattr(args, "input_type", None),
            getattr(args, "output_type", None),
        )
    elif args.command == "generate-config":
        return generate_config_command(
            skeleton=getattr(args, "skeleton", None),
            generate_all=getattr(args, "all", False),
            output=getattr(args, "output", None),
            output_dir=getattr(args, "output_dir", None),
            show_alternatives=getattr(args, "show_alternatives", False),
        )
    
    return 1


def _resolve_app_module(app_path: str) -> str:
    """Convert app path to module name.
    
    Examples:
        apps/algo-trade -> algo_trade_dag
        apps/pipeline-app -> pipeline_app
    """
    path = Path(app_path)
    app_name = path.name.replace("-", "_")
    return f"{app_name}_dag"


if __name__ == "__main__":
    sys.exit(main())
```

**使用例（xform_auditorと同じパターン）**:
```bash
# Generate sample config from skeleton (NEW!)
uv run python -m xform_core.dag apps/algo-trade generate-config \
  --skeleton phase1_skeleton \
  --output configs/samples/phase1_sample.yaml

# Validate configuration
uv run python -m xform_core.dag apps/algo-trade validate configs/my_pipeline.yaml

# Run pipeline
uv run python -m xform_core.dag apps/algo-trade run configs/my_pipeline.yaml

# Discover transforms
uv run python -m xform_core.dag apps/algo-trade discover --input-type MarketDataIngestionConfig

# 別のアプリでも同じインターフェース
uv run python -m xform_core.dag apps/pipeline-app validate configs/test.yaml
```

**App側は CLIコード不要**: `skeleton.py` を定義するだけで、自動的に Core CLI から利用可能になり、サンプル設定も自動生成できます。

### 3. Configuration Validator (Core)

**Location**: `packages/xform-core/xform_core/dag/validator.py`

**Critical Principle**: 設定ファイルの全エラーを**パイプライン実行前**に検出し、Fail Fast する。

```python
from typing import Any
from dataclasses import dataclass
from inspect import signature, Parameter
from xform_core.transform_registry import TransformRegistry
from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep

@dataclass
class ValidationError:
    """Single validation error."""
    phase: str
    step: str
    error_type: str
    message: str
    suggestion: str | None = None

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]
    
    def __str__(self) -> str:
        """Format validation result for display."""
        if self.is_valid:
            return "✓ Configuration is valid"
        
        lines = ["✗ Configuration validation failed:\n"]
        
        for err in self.errors:
            lines.append(f"  [ERROR] {err.phase}.{err.step}: {err.error_type}")
            lines.append(f"    {err.message}")
            if err.suggestion:
                lines.append(f"    Suggestion: {err.suggestion}")
            lines.append("")
        
        for warn in self.warnings:
            lines.append(f"  [WARNING] {warn.phase}.{warn.step}: {warn.error_type}")
            lines.append(f"    {warn.message}")
            lines.append("")
        
        return "\n".join(lines)

class ConfigurationValidator:
    """Validate pipeline configuration before execution."""
    
    def __init__(
        self,
        registry: TransformRegistry,
        skeleton: PipelineSkeleton,
    ):
        self.registry = registry
        self.skeleton = skeleton
    
    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate entire configuration.
        
        Checks performed:
        1. Skeleton step coverage (all required steps present)
        2. Transform FQN existence
        3. Type signature compatibility
        4. Parameter schema validation
        5. Required parameter completeness
        
        Returns:
            ValidationResult with all errors and warnings
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        
        # Extract phase name from skeleton
        phase_name = self.skeleton.name
        
        # Check 1: Skeleton step coverage
        config_steps = set(config.get("steps", {}).keys())
        required_steps = {
            step.name for step in self.skeleton.steps if step.required
        }
        missing_steps = required_steps - config_steps
        
        for step_name in missing_steps:
            step = next(s for s in self.skeleton.steps if s.name == step_name)
            if step.default_transform is None:
                errors.append(ValidationError(
                    phase=phase_name,
                    step=step_name,
                    error_type="MISSING_REQUIRED_STEP",
                    message=f"Required step '{step_name}' not found in configuration",
                    suggestion=f"Add step configuration with transform selection",
                ))
        
        # Check each configured step
        for step_name, step_config in config.get("steps", {}).items():
            # Find step in skeleton
            step = next(
                (s for s in self.skeleton.steps if s.name == step_name),
                None,
            )
            
            if step is None:
                warnings.append(ValidationError(
                    phase=phase_name,
                    step=step_name,
                    error_type="UNKNOWN_STEP",
                    message=f"Step '{step_name}' not defined in skeleton",
                    suggestion="Remove this step or check skeleton definition",
                ))
                continue
            
            # Get transform FQN
            transform_fqn = step_config.get("transform", step.default_transform)
            
            if transform_fqn is None:
                errors.append(ValidationError(
                    phase=phase_name,
                    step=step_name,
                    error_type="NO_TRANSFORM_SPECIFIED",
                    message=f"No transform specified and no default available",
                    suggestion=self._suggest_transforms(step),
                ))
                continue
            
            # Check 2: Transform existence
            if not self.registry.has_transform(transform_fqn):
                errors.append(ValidationError(
                    phase=phase_name,
                    step=step_name,
                    error_type="TRANSFORM_NOT_FOUND",
                    message=f"Transform '{transform_fqn}' not found in registry",
                    suggestion=self._suggest_transforms(step),
                ))
                continue
            
            # Check 3: Type signature compatibility
            if not self.registry.validate_signature(
                transform_fqn,
                step.input_types,
                step.output_type,
            ):
                actual_sig = self.registry.get_signature(transform_fqn)
                errors.append(ValidationError(
                    phase=phase_name,
                    step=step_name,
                    error_type="TYPE_SIGNATURE_MISMATCH",
                    message=(
                        f"Transform signature mismatch:\n"
                        f"  Expected: {step.input_types} -> {step.output_type}\n"
                        f"  Actual: {actual_sig.input_types} -> {actual_sig.output_type}"
                    ),
                    suggestion=self._suggest_transforms(step),
                ))
                continue
            
            # Check 4 & 5: Parameter validation
            params = step_config.get("params", {})
            param_errors = self._validate_parameters(
                phase_name,
                step_name,
                transform_fqn,
                params,
            )
            errors.extend(param_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _validate_parameters(
        self,
        phase: str,
        step: str,
        transform_fqn: str,
        params: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate parameters against transform signature."""
        errors: list[ValidationError] = []
        
        # Get function signature
        func = self.registry.get_transform(transform_fqn)
        sig = signature(func)
        
        # Check for unknown parameters
        valid_params = {
            name for name, param in sig.parameters.items()
            if param.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            and name != "self"  # Exclude first positional arg (input data)
        }
        
        unknown_params = set(params.keys()) - valid_params
        for param_name in unknown_params:
            errors.append(ValidationError(
                phase=phase,
                step=step,
                error_type="UNKNOWN_PARAMETER",
                message=f"Parameter '{param_name}' not accepted by {transform_fqn}",
                suggestion=f"Valid parameters: {', '.join(sorted(valid_params))}",
            ))
        
        # Check for missing required parameters
        required_params = {
            name for name, param in sig.parameters.items()
            if param.default is Parameter.empty
            and param.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            and name not in ("self", list(sig.parameters.keys())[0])  # Exclude self and first arg
        }
        
        missing_params = required_params - set(params.keys())
        for param_name in missing_params:
            errors.append(ValidationError(
                phase=phase,
                step=step,
                error_type="MISSING_REQUIRED_PARAMETER",
                message=f"Required parameter '{param_name}' not provided",
                suggestion=f"Add '{param_name}' to params section",
            ))
        
        # Type validation (basic)
        for param_name, param_value in params.items():
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.annotation is not Parameter.empty:
                    # Basic type checking
                    expected_type = param.annotation
                    # Handle Optional, Union, etc. (simplified)
                    if hasattr(expected_type, "__origin__"):
                        # Skip complex generics for now
                        continue
                    
                    if not isinstance(param_value, expected_type):
                        errors.append(ValidationError(
                            phase=phase,
                            step=step,
                            error_type="PARAMETER_TYPE_MISMATCH",
                            message=(
                                f"Parameter '{param_name}' type mismatch:\n"
                                f"  Expected: {expected_type.__name__}\n"
                                f"  Got: {type(param_value).__name__}"
                            ),
                            suggestion=f"Convert value to {expected_type.__name__}",
                        ))
        
        return errors
    
    def _suggest_transforms(self, step: PipelineStep) -> str:
        """Suggest available transforms for a step."""
        candidates = self.registry.find_transforms(
            step.input_types,
            step.output_type,
        )
        
        if not candidates:
            return f"No compatible transforms found for {step.input_types} -> {step.output_type}"
        
        return f"Available transforms:\n    " + "\n    ".join(f"- {c}" for c in candidates)
```

### 4. Transform Resolver (Core)

**Location**: `packages/xform-core/xform_core/dag/resolver.py`

```python
from typing import Callable, Any
from xform_core.transform_registry import TransformRegistry

class TransformResolver:
    """Resolve transform functions from configuration.
    
    Note: This assumes configuration has already been validated.
    Use ConfigurationValidator before calling resolver methods.
    """
    
    def __init__(self, registry: TransformRegistry):
        self.registry = registry
    
    def resolve_step(
        self,
        step: PipelineStep,
        config: dict[str, Any],
    ) -> tuple[Callable, dict[str, Any]]:
        """Resolve transform function and parameters for a step.
        
        Returns:
            Tuple of (function, params)
        
        Raises:
            ValueError: If no transform specified
            TypeError: If type signature mismatch (should be caught by validator)
        """
        # Get transform FQN from config or use default
        transform_fqn = config.get("transform", step.default_transform)
        if transform_fqn is None:
            raise ValueError(f"No transform specified for step: {step.name}")
        
        # Get actual function (validation already done)
        func = self.registry.get_transform(transform_fqn)
        
        # Get parameters
        params = config.get("params", {})
        
        return func, params
```

### 5. DAG Executor (Core)

**Location**: `packages/xform-core/xform_core/dag/executor.py`

```python
from typing import Any, Type
from xform_core.dag.skeleton import PipelineSkeleton
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.validator import ConfigurationValidator, ValidationResult

class DAGExecutor:
    """Execute pipeline with dynamic transform selection.
    
    CRITICAL: Configuration must be validated before execution.
    """
    
    def __init__(
        self,
        skeleton: PipelineSkeleton,
        resolver: TransformResolver,
        validator: ConfigurationValidator,
    ):
        self.skeleton = skeleton
        self.resolver = resolver
        self.validator = validator
    
    def execute(
        self,
        config: dict[str, Any],
        initial_inputs: dict[str, Any],
        *,
        skip_validation: bool = False,
    ) -> dict[str, Any]:
        """Execute pipeline with configuration.
        
        Parameters:
            config: Step configurations with transform selections
            initial_inputs: Initial data for pipeline
            skip_validation: Skip validation (NOT RECOMMENDED, for testing only)
        
        Returns:
            Dictionary of step outputs
        
        Raises:
            ValueError: If configuration is invalid
        """
        # CRITICAL: Validate configuration before execution
        if not skip_validation:
            validation_result = self.validator.validate(config)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Configuration validation failed:\n{validation_result}"
                )
        
        outputs = {}
        context = {**initial_inputs}
        
        for step in self.skeleton.steps:
            # Get step configuration
            step_config = config.get("steps", {}).get(step.name, {})
            
            # Skip optional steps if not configured
            if not step_config and not step.required:
                continue
            
            # Resolve transform function and parameters
            func, params = self.resolver.resolve_step(step, step_config)
            
            # Collect inputs from context
            inputs = []
            for input_type in step.input_types:
                input_data = self._find_input_by_type(context, input_type)
                if input_data is None:
                    raise RuntimeError(
                        f"Required input type {input_type} not found in context "
                        f"for step {step.name}"
                    )
                inputs.append(input_data)
            
            # Execute transform
            print(f"Executing step: {step.name} with {func.__name__}")
            output = func(*inputs, **params)
            
            # Store output in context
            outputs[step.name] = output
            context[step.output_type.__name__] = output
        
        return outputs
    
    def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate configuration without executing pipeline.
        
        This should be called explicitly when loading configuration.
        """
        return self.validator.validate(config)
    
    def _find_input_by_type(
        self,
        context: dict[str, Any],
        target_type: Type[Any],
    ) -> Any | None:
        """Find data in context matching target type.
        
        Type matching strategy:
        1. Exact type name match
        2. Instance type check
        3. Structural compatibility (duck typing)
        """
        # Strategy 1: Exact type name match
        type_name = target_type.__name__
        if type_name in context:
            return context[type_name]
        
        # Strategy 2: Instance type check
        for value in context.values():
            if isinstance(value, target_type):
                return value
        
        # Strategy 3: Check for TypedDict or structural types
        # (Implementation depends on runtime type checking library)
        # For now, return None if not found
        return None
```

## Configuration Format

### Enhanced YAML Structure

```yaml
pipeline:
  name: "algo-trade-dynamic-pipeline"
  version: "2.0"
  skeleton: "algo_trade_dag.skeleton.full_pipeline_skeleton"

phase1:
  steps:
    fetch_yahoo:
      transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"
      params:
        use_adjusted_close: true
    
    fetch_ccxt:
      transform: "algo_trade_transforms.market_data.fetch_ccxt_ohlcv"
      params:
        rate_limit_ms: 1000
    
    normalize:
      transform: "algo_trade_transforms.market_data.normalize_multi_provider"
      params:
        target_frequency: "1H"
    
    merge:
      # Use default from skeleton
      params:
        interpolation_method: "linear"
    
    persist:
      transform: "algo_trade_transforms.market_data.persist_market_data_snapshot"
      params:
        base_dir: "output/data/snapshots"

phase2:
  steps:
    load_data:
      transform: "algo_trade_transforms.market_data.load_market_data"
    
    calculate_indicators:
      # Parallel indicator calculation (fan-out pattern)
      parallel:
        - transform: "algo_trade_transforms.transforms.calculate_rsi"
          params:
            period: 14
        - transform: "algo_trade_transforms.transforms.calculate_adx"
          params:
            period: 14
        - transform: "algo_trade_transforms.transforms.calculate_volatility"
          params:
            window: 20
    
    calculate_target:
      transform: "algo_trade_transforms.transforms.calculate_future_return"
      params:
        forward: 5
    
    select_features:
      transform: "algo_trade_transforms.transforms.select_features"
      params:
        feature_specs:
          - ["USDJPY", "rsi", 14]
          - ["USDJPY", "adx", 14]
    
    extract_target:
      transform: "algo_trade_transforms.transforms.extract_target"
      params:
        symbol: "USDJPY"
        indicator: "future_return"
    
    align:
      transform: "algo_trade_transforms.transforms.clean_and_align"
```

## Configuration Validation CLI

### Validate Command (CRITICAL - Must Run Before Execution)

```bash
# Validate pipeline configuration (REQUIRED before execution)
uv run python -m xform_core.dag apps/algo-trade validate configs/my_pipeline.yaml

# Success output:
# ✓ Configuration is valid
# 
# All checks passed:
#   - Skeleton step coverage: ✓
#   - Transform FQN existence: ✓
#   - Type signature compatibility: ✓
#   - Parameter schema validation: ✓
#   - Required parameter completeness: ✓

# Error output example:
# ✗ Configuration validation failed:
#
#   [ERROR] phase1.fetch_data: TRANSFORM_NOT_FOUND
#     Transform 'algo_trade_transforms.market_data.fetch_invalid_ohlcv' not found in registry
#     Suggestion: Available transforms:
#       - algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv
#       - algo_trade_transforms.market_data.fetch_ccxt_ohlcv
#
#   [ERROR] phase2.calculate_rsi: TYPE_SIGNATURE_MISMATCH
#     Transform signature mismatch:
#       Expected: (MultiAssetOHLCVFrame,) -> MultiAssetOHLCVFrame
#       Actual: (pd.DataFrame,) -> pd.Series
#     Suggestion: Available transforms:
#       - algo_trade_transforms.transforms.calculate_rsi
#
#   [ERROR] phase2.calculate_rsi: MISSING_REQUIRED_PARAMETER
#     Required parameter 'period' not provided
#     Suggestion: Add 'period' to params section
#
#   [WARNING] phase3.custom_step: UNKNOWN_STEP
#     Step 'custom_step' not defined in skeleton
#     Suggestion: Remove this step or check skeleton definition

# Validate for different apps using the same interface
uv run python -m xform_core.dag apps/pipeline-app validate configs/test.yaml
```

### Validation Error Types

| Error Type | Description | Severity |
|-----------|-------------|----------|
| **MISSING_REQUIRED_STEP** | Required step not configured | ERROR |
| **NO_TRANSFORM_SPECIFIED** | No transform FQN and no default | ERROR |
| **TRANSFORM_NOT_FOUND** | Transform FQN not in registry | ERROR |
| **TYPE_SIGNATURE_MISMATCH** | Transform types don't match skeleton | ERROR |
| **UNKNOWN_PARAMETER** | Parameter not accepted by function | ERROR |
| **MISSING_REQUIRED_PARAMETER** | Required param not provided | ERROR |
| **PARAMETER_TYPE_MISMATCH** | Parameter value type incorrect | ERROR |
| **UNKNOWN_STEP** | Step not in skeleton | WARNING |

### Configuration Generation CLI

Skeleton定義からサンプル設定ファイルを自動生成：

```bash
# Generate config from skeleton
uv run python -m xform_core.dag apps/algo-trade generate-config \
  --skeleton phase1_skeleton \
  --output configs/samples/phase1_sample.yaml

# Output: configs/samples/phase1_sample.yaml
# pipeline:
#   name: "phase1_market_data_ingestion"
#   skeleton: "algo_trade_dag.skeleton.phase1_skeleton"
# 
# phase1:
#   steps:
#     fetch_data:
#       transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"
#       params:
#         use_adjusted_close: true  # Default from signature
#     normalize:
#       transform: "algo_trade_transforms.market_data.normalize_multi_provider"
#       params:
#         target_frequency: "1H"  # Default from signature
#     # ... all steps with default transforms and parameters

# Generate all skeletons at once
uv run python -m xform_core.dag apps/algo-trade generate-config --all \
  --output-dir configs/samples/

# Generate with alternative transforms (show candidates)
uv run python -m xform_core.dag apps/algo-trade generate-config \
  --skeleton phase1_skeleton \
  --show-alternatives \
  --output configs/samples/phase1_with_alternatives.yaml

# Output includes comments with alternative transforms:
# phase1:
#   steps:
#     fetch_data:
#       transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"
#       # Alternatives:
#       #   - algo_trade_transforms.market_data.fetch_ccxt_ohlcv
#       #   - algo_trade_transforms.market_data.fetch_mock_ohlcv
#       params:
#         use_adjusted_close: true
```

### Transform Discovery CLI

```bash
# List available transforms for specific type conversion
uv run python -m xform_core.dag apps/algo-trade discover \
  --input-type MarketDataIngestionConfig \
  --output-type ProviderBatchCollection

# Output:
# Available transforms for MarketDataIngestionConfig -> ProviderBatchCollection:
#   - algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv
#       Parameters: use_adjusted_close (bool), rate_limit_ms (int)
#   - algo_trade_transforms.market_data.fetch_ccxt_ohlcv
#       Parameters: rate_limit_ms (int), exchange (str)

# List all transforms in registry
uv run python -m xform_core.dag apps/algo-trade discover

# Same interface works for all apps
uv run python -m xform_core.dag apps/pipeline-app discover
```

### Integration with DAG Execution

**Core Configuration Loader** (`packages/xform-core/xform_core/dag/config.py`):

```python
"""Core configuration loading with validation."""

import yaml
from pathlib import Path
from typing import Any
from xform_core.dag.validator import ConfigurationValidator, ValidationResult
from xform_core.dag.skeleton import get_skeleton
from xform_core.transform_registry import get_registry

class PipelineConfig:
    """Pipeline configuration with automatic validation."""
    
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        
        # Load YAML
        with open(self.config_path) as f:
            self._raw = yaml.safe_load(f)
        
        # Get skeleton
        skeleton_fqn = self._raw.get("pipeline", {}).get("skeleton")
        self.skeleton = get_skeleton(skeleton_fqn)
        
        # Validate IMMEDIATELY on load
        registry = get_registry()
        validator = ConfigurationValidator(registry, self.skeleton)
        self.validation_result = validator.validate(self._raw)
        
        # FAIL FAST if invalid
        if not self.validation_result.is_valid:
            raise ValueError(
                f"Configuration validation failed for {config_path}:\n"
                f"{self.validation_result}"
            )
    
    def get_phase_config(self, phase: str) -> dict[str, Any]:
        """Get configuration for a specific phase."""
        return self._raw.get(phase, {})

# App-specific usage (apps/algo-trade/algo_trade_dag/run.py)
def run_pipeline_from_config(config_path: str) -> dict:
    """Run algo-trade pipeline from configuration file.
    
    Configuration is validated automatically on load (using core PipelineConfig).
    If validation fails, ValueError is raised before execution starts.
    """
    from xform_core.dag.config import PipelineConfig
    from algo_trade_dag.skeleton import phase1_skeleton, phase2_skeleton  # Import to register
    
    # This will validate and raise if invalid
    config = PipelineConfig(config_path)
    
    # If we reach here, config is guaranteed to be valid
    return run_full_pipeline(
        ingestion_config=config.get_phase_config("phase1"),
        feature_specs=config.get_phase_config("phase2").get("feature_specs"),
        # ...
    )
```

**Note**: アプリ固有のSkeleton定義（`apps/algo-trade/algo_trade_dag/skeleton.py`）を定義すると、Core CLIが自動的にimportして登録する。

## アーキテクチャの核心

この設計は以下の3層で構成される：

### 1. Skeleton（型変換フロー定義）
```python
# App側: 型シグネチャだけを定義
phase1_skeleton = PipelineSkeleton(
    name="phase1_market_data_ingestion",
    steps=[
        PipelineStep(
            name="fetch_data",
            input_types=(MarketDataIngestionConfig,),
            output_type=ProviderBatchCollection,
        ),
        # 実装は指定しない
    ],
)
```

### 2. Configuration（Transform選択）
```yaml
# YAML: 実装とパラメータを外部化
phase1:
  steps:
    fetch_data:
      transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"
      params:
        use_adjusted_close: true
```

### 3. Execution（自動実行）
```bash
# Core CLIが全て処理
uv run python -m xform_core.dag apps/algo-trade run configs/pipeline.yaml
```

### コード量の比較

| Component | Code Required |
|-----------|--------------|
| Skeleton定義 | 100行（型フローのみ） |
| CLIコード | **0行（Core提供）** |
| DAG実装 | **0行（Executor提供）** |
| **合計** | **100行** |

**従来の手続き型DAGとの比較**: 詳細は [`doc/MIGRATION_TO_DYNAMIC_DAG.md`](../MIGRATION_TO_DYNAMIC_DAG.md) を参照。

## 実装優先度

実装とテストを統合した段階的な開発計画。テスト仕様の詳細は [`doc/dynamic-pipeline/dag-dynamic-selection-test-spec.md`](../../dynamic-pipeline/dag-dynamic-selection-test-spec.md) を参照。

### Phase 0: テストインフラ準備（Setup Test Infrastructure）

**実装**:
- ⬜ Create `packages/xform-core/tests/dag/` directory structure
- ⬜ Create `fixtures/test_skeleton.py` using pipeline-app transforms
- ⬜ Implement `conftest.py` with fixtures:
  - `import_pipeline_app` (auto-registers transforms)
  - `test_registry` (populated with pipeline-app)
  - `test_skeleton_registered`
  - `generated_config_dir` (tmp_path based)
  - `valid_config_path` (uses `generate-config` CLI)
  - `invalid_config_*` fixtures (programmatic creation)

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/conftest.py --collect-only
```

**完了条件**:
- ✅ Fixture が正常にimportできる
- ✅ pipeline-app transforms が自動登録される
- ✅ Test skeleton が登録できる

**所要時間**: 1-2時間

---

### Phase 1: Enhanced Transform Registry（実装 + テスト）

**実装** (`packages/xform-core/xform_core/transform_registry.py`):
1. ⬜ `TransformSignature` dataclass (input_types, output_type, params)
2. ⬜ `TransformRegistry` class:
   - `register(fqn, func, signature)` - FQN と型シグネチャで登録
   - `find_transforms(input_types, output_type)` - 型ベース検索
   - `get_transform(fqn)` - FQN から関数解決
   - `validate_signature(fqn, input_types, output_type)` - 型互換性検証
   - `has_transform(fqn)` - 存在チェック
   - `get_signature(fqn)` - シグネチャ取得
3. ⬜ Global registry instance (`get_registry()`)

**テスト** (`packages/xform-core/tests/dag/test_transform_registry_enhanced.py`):
- ⬜ TR-N-01: Register → Find → Get → Validate (統合型正常系)
- ⬜ TR-E-01: Get with non-existent FQN
- ⬜ TR-E-02: Validate signature with type mismatch

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_transform_registry_enhanced.py -v
```

**完了条件**:
- ✅ 3 tests pass
- ✅ pipeline-app transforms が正常に登録・検索できる
- ✅ Coverage 80%+

**所要時間**: 3-4時間

---

### Phase 2: DAG Skeleton Definition（実装 + テスト）

**実装** (`packages/xform-core/xform_core/dag/skeleton.py`):
1. ⬜ `PipelineStep` dataclass (name, input_types, output_type, default_transform, required)
2. ⬜ `PipelineSkeleton` class:
   - `validate_config(config)` - 設定の完全性チェック
3. ⬜ Skeleton registry:
   - `_SKELETON_REGISTRY` (dict)
   - `register_skeleton(fqn, skeleton)`
   - `get_skeleton(fqn)`

**テスト** (`packages/xform-core/tests/dag/test_dag_skeleton.py`):
- ⬜ SK-N-01: Create → Register → Get → Validate config (統合型正常系)
- ⬜ SK-E-01: Get skeleton with non-existent FQN

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_dag_skeleton.py -v
```

**完了条件**:
- ✅ 2 tests pass
- ✅ Skeleton 定義と検証ロジックが動作
- ✅ Coverage 75%+

**所要時間**: 2-3時間

---

### Phase 3: Configuration Validator（実装 + テスト）**[CRITICAL]**

**実装** (`packages/xform-core/xform_core/dag/validator.py`):
1. ⬜ `ValidationError` dataclass (phase, step, error_type, message, suggestion)
2. ⬜ `ValidationResult` dataclass (is_valid, errors, warnings) + `__str__` formatting
3. ⬜ `ConfigurationValidator` class:
   - `validate(config)` - 全チェックを実行
   - `_validate_parameters(...)` - パラメータ検証
   - `_suggest_transforms(step)` - Transform候補提案
4. ⬜ Error type detection:
   - MISSING_REQUIRED_STEP
   - NO_TRANSFORM_SPECIFIED
   - TRANSFORM_NOT_FOUND
   - TYPE_SIGNATURE_MISMATCH
   - UNKNOWN_PARAMETER
   - MISSING_REQUIRED_PARAMETER
   - PARAMETER_TYPE_MISMATCH
   - UNKNOWN_STEP (warning)

**テスト** (`packages/xform-core/tests/dag/test_configuration_validator.py`):

**正常系 (4 tests)**:
- ⬜ CV-N-01: Complete valid configuration
- ⬜ CV-N-02: Config using default transforms
- ⬜ CV-N-03: Config with all optional parameters
- ⬜ CV-N-04: Config with correct type annotations

**異常系 (8 tests)** - 各エラータイプを個別に検証:
- ⬜ CV-E-01: MISSING_REQUIRED_STEP
- ⬜ CV-E-02: NO_TRANSFORM_SPECIFIED
- ⬜ CV-E-03: TRANSFORM_NOT_FOUND
- ⬜ CV-E-04: TYPE_SIGNATURE_MISMATCH (input)
- ⬜ CV-E-05: TYPE_SIGNATURE_MISMATCH (output)
- ⬜ CV-E-06: UNKNOWN_PARAMETER
- ⬜ CV-E-07: MISSING_REQUIRED_PARAMETER
- ⬜ CV-E-08: PARAMETER_TYPE_MISMATCH

**警告系 (1 test)**:
- ⬜ CV-W-01: UNKNOWN_STEP

**複合系 (2 tests)**:
- ⬜ CV-M-01: Multiple errors (2+ types)
- ⬜ CV-M-02: Errors and warnings

**Suggestion品質 (3 tests)**:
- ⬜ CV-S-01: TRANSFORM_NOT_FOUND includes suggestions
- ⬜ CV-S-02: UNKNOWN_PARAMETER includes valid params
- ⬜ CV-S-03: TYPE_SIGNATURE_MISMATCH includes alternatives

**検証**:
```bash
# All validator tests (17 tests)
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py -v

# Verify all error types are covered
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py -v -k "CV_E"
```

**完了条件**:
- ✅ 17 tests pass (4 + 8 + 1 + 2 + 3)
- ✅ 全8種類のエラータイプが検出される
- ✅ Suggestion が適切に生成される
- ✅ Coverage 95%+

**所要時間**: 6-8時間（最も重要なコンポーネント）

---

### Phase 4: Transform Resolver & DAG Executor（実装 + テスト）

**実装 - Resolver** (`packages/xform-core/xform_core/dag/resolver.py`):
1. ⬜ `TransformResolver` class:
   - `resolve_step(step, config)` - Transform + params 解決

**実装 - Executor** (`packages/xform-core/xform_core/dag/executor.py`):
1. ⬜ `DAGExecutor` class:
   - `execute(config, initial_inputs, skip_validation=False)` - パイプライン実行
   - `validate_config(config)` - 明示的なバリデーション
   - `_find_input_by_type(context, target_type)` - 型ベース入力検索

**テスト - Resolver** (`packages/xform-core/tests/dag/test_transform_resolver.py`):
- ⬜ RS-N-01: Resolve with explicit transform + params
- ⬜ RS-N-02: Resolve using default transform
- ⬜ RS-E-01: Resolve with no transform and no default

**テスト - Executor** (`packages/xform-core/tests/dag/test_dag_executor.py`):
- ⬜ EX-N-01: Execute multi-step pipeline with validation
- ⬜ EX-E-01: Execute with invalid config (Fail Fast)
- ⬜ EX-E-02: Execute with missing required input

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_transform_resolver.py -v
uv run pytest packages/xform-core/tests/dag/test_dag_executor.py -v
```

**完了条件**:
- ✅ 6 tests pass (3 + 3)
- ✅ Fail Fast が機能（EX-E-01）
- ✅ Coverage 80%+

**所要時間**: 4-5時間

---

### Phase 5: Configuration Loading（実装 + テスト）

**実装** (`packages/xform-core/xform_core/dag/config.py`):
1. ⬜ `PipelineConfig` class:
   - `__init__(config_path)` - YAML load + auto-validation (Fail Fast)
   - `get_phase_config(phase)` - Phase別設定取得

**テスト** (`packages/xform-core/tests/dag/test_pipeline_config.py`):
- ⬜ CFG-N-01: Load valid YAML → auto-validate → get phase config
- ⬜ CFG-E-01: Load config with validation errors (Fail Fast)
- ⬜ CFG-E-02: Load non-existent config file

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_pipeline_config.py -v
```

**完了条件**:
- ✅ 3 tests pass
- ✅ Invalid config で即座に ValueError
- ✅ Coverage 75%+

**所要時間**: 2-3時間

---

### Phase 6: CLI Commands（実装 + テスト）

**実装** (`packages/xform-core/xform_core/dag/cli.py`):
1. ⬜ `validate_command(config_path)` - 設定検証
2. ⬜ `run_command(config_path, initial_inputs)` - パイプライン実行
3. ⬜ `discover_command(input_type, output_type)` - Transform検索
4. ⬜ `generate_config_command(...)` - 設定自動生成
   - Single skeleton generation
   - Bulk generation (`--all`)
   - Parameter extraction from signatures
   - Alternative transforms as comments (`--show-alternatives`)

**テスト** (`packages/xform-core/tests/dag/test_dag_cli.py`):
- ⬜ CLI-N-01: validate_command with valid config
- ⬜ CLI-N-02: run_command with valid config
- ⬜ CLI-N-03: generate_config_command for single skeleton
- ⬜ CLI-E-01: validate_command with invalid config
- ⬜ CLI-E-02: run_command with invalid config (Fail Fast)

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_dag_cli.py -v
```

**完了条件**:
- ✅ 5 tests pass
- ✅ `generate-config` が動作する YAML を生成
- ✅ Coverage 70%+

**所要時間**: 4-5時間

---

### Phase 7: Unified CLI Entry Point（実装 + テスト）

**実装** (`packages/xform-core/xform_core/dag/__main__.py`):
1. ⬜ Argument parser setup (app_path, commands)
2. ⬜ `_resolve_app_module(app_path)` - App path → module name
3. ⬜ Dynamic skeleton import (`import_module(f"{app_module}.skeleton")`)
4. ⬜ Command dispatch (validate, run, discover, generate-config)

**テスト** (`packages/xform-core/tests/dag/test_dag_main.py`):
- ⬜ MAIN-N-01: Run with valid app_path → skeleton auto-registered
- ⬜ MAIN-E-01: Run with non-existent app_path

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_dag_main.py -v

# Manual CLI test
uv run python -m xform_core.dag apps/pipeline-app generate-config \
  --skeleton test_pipeline_skeleton \
  --output /tmp/test.yaml
```

**完了条件**:
- ✅ 2 tests pass
- ✅ CLI が実際に動作する
- ✅ Coverage 70%+

**所要時間**: 3-4時間

---

### Phase 8: Integration Tests（E2E検証）

**実装** (`packages/xform-core/tests/dag/test_dag_integration.py`):
- ⬜ INT-E2E-01: Full workflow (Define skeleton → Generate config → Validate → Run)
  - Step 1: `generate-config` CLI でYAML生成
  - Step 2: 生成された設定を検証（内容確認）
  - Step 3: `validate` CLI で検証
  - Step 4: `run` CLI でパイプライン実行
  - 全ステップの成功を確認

**検証**:
```bash
uv run pytest packages/xform-core/tests/dag/test_dag_integration.py -v

# Manual E2E verification
uv run python -m xform_core.dag apps/pipeline-app generate-config \
  --skeleton test_pipeline_skeleton --output /tmp/test.yaml
uv run python -m xform_core.dag apps/pipeline-app validate /tmp/test.yaml
uv run python -m xform_core.dag apps/pipeline-app run /tmp/test.yaml
```

**完了条件**:
- ✅ 1 test pass
- ✅ Full E2E workflow が動作
- ✅ 手動実行でも成功

**所要時間**: 2-3時間

---

### Phase 9: App-Specific Implementation（algo-trade）

**実装** (`apps/algo-trade/algo_trade_dag/skeleton.py`):
1. ⬜ Phase1 Skeleton (Market Data Ingestion)
   - fetch_data, normalize, merge, persist steps
2. ⬜ Phase2 Skeleton (Feature Engineering)
   - load_data, calculate_indicators, calculate_target, select_features, extract_target, align steps
3. ⬜ Phase3 Skeleton (Training & Prediction)
   - cv_split, train_model, predict steps
4. ⬜ Phase4 Skeleton (Simulation)
   - simulate, analyze steps
5. ⬜ Register all skeletons with FQN

**検証**:
```bash
# Generate sample configs for all phases
uv run python -m xform_core.dag apps/algo-trade generate-config --all \
  --output-dir configs/samples/

# Validate each phase config
uv run python -m xform_core.dag apps/algo-trade validate configs/samples/phase1_market_data_ingestion.yaml
uv run python -m xform_core.dag apps/algo-trade validate configs/samples/phase2_feature_engineering.yaml
uv run python -m xform_core.dag apps/algo-trade validate configs/samples/phase3_training_prediction.yaml
uv run python -m xform_core.dag apps/algo-trade validate configs/samples/phase4_simulation.yaml
```

**完了条件**:
- ✅ 4つの Skeleton 定義完了（Phase1-4）
- ✅ 各 Skeleton から設定ファイル自動生成
- ✅ 全設定ファイルが検証通過

**所要時間**: 3-4時間

**Note**: サンプル設定ファイルは `generate-config` コマンドで自動生成されます。手動作成は不要です。

---

## 実装進行状況の確認

### 全体テスト実行
```bash
# All DAG tests (39 tests expected)
uv run pytest packages/xform-core/tests/dag/ -v

# Test count verification
uv run pytest packages/xform-core/tests/dag/ --collect-only | grep "test session"

# Coverage report
uv run pytest packages/xform-core/tests/dag/ \
  --cov=xform_core.dag \
  --cov-report=term-missing \
  --cov-report=html:output/coverage/dag
```

### Phase別進捗チェック

```bash
# Phase 1: Registry (3 tests)
uv run pytest packages/xform-core/tests/dag/test_transform_registry_enhanced.py -v

# Phase 2: Skeleton (2 tests)
uv run pytest packages/xform-core/tests/dag/test_dag_skeleton.py -v

# Phase 3: Validator (17 tests) - CRITICAL
uv run pytest packages/xform-core/tests/dag/test_configuration_validator.py -v

# Phase 4: Resolver + Executor (6 tests)
uv run pytest packages/xform-core/tests/dag/test_transform_resolver.py -v
uv run pytest packages/xform-core/tests/dag/test_dag_executor.py -v

# Phase 5: Config Loading (3 tests)
uv run pytest packages/xform-core/tests/dag/test_pipeline_config.py -v

# Phase 6: CLI Commands (5 tests)
uv run pytest packages/xform-core/tests/dag/test_dag_cli.py -v

# Phase 7: Unified CLI (2 tests)
uv run pytest packages/xform-core/tests/dag/test_dag_main.py -v

# Phase 8: Integration (1 test)
uv run pytest packages/xform-core/tests/dag/test_dag_integration.py -v
```

### 完了基準

**Must Pass (23 tests - Blocking)**:
- ✅ Configuration Validator: 17 tests (全エラータイプ)
- ✅ Fail Fast Validation: 3 tests (EX-E-01, CFG-E-01, CLI-E-02)
- ✅ E2E Integration: 1 test (INT-E2E-01)
- ✅ Other Components Happy Path: 2 tests (各コンポーネント正常系)

**Should Pass (16 tests - High Priority)**:
- ✅ All remaining normal cases
- ✅ All remaining error cases

**Total**: 39 tests

---

## 開発時間見積もり

| Phase | Component | Implementation | Testing | Total | Priority |
|-------|-----------|----------------|---------|-------|----------|
| 0 | Test Infrastructure | 1-2h | - | 1-2h | Setup |
| 1 | Transform Registry | 2h | 1-2h | 3-4h | High |
| 2 | Skeleton Definition | 1-2h | 1h | 2-3h | High |
| 3 | **Configuration Validator** | **4h** | **2-4h** | **6-8h** | **CRITICAL** |
| 4 | Resolver + Executor | 3h | 1-2h | 4-5h | High |
| 5 | Configuration Loading | 1-2h | 1h | 2-3h | Medium |
| 6 | CLI Commands | 2-3h | 1-2h | 4-5h | High |
| 7 | Unified CLI | 2h | 1-2h | 3-4h | High |
| 8 | Integration Tests | - | 2-3h | 2-3h | Validation |
| 9 | App Implementation | 3-4h | - | 3-4h | App-specific |
| **Total** | | **19-24h** | **11-18h** | **30-42h** | |

**推奨アプローチ**: Phase 0-8 を順次実装（Core基盤完成） → Phase 9（App実装）

**Critical Path**: Phase 3 (Configuration Validator) が最も重要。他のコンポーネントの品質を保証する基盤となる。


## Benefits

1. **Fail-Fast Validation**: 設定エラーを実行前に完全検出、デバッグ時間を大幅削減
2. **Type Safety**: 型シグネチャの検証により、実行時エラーを事前に防止
3. **Flexibility**: 型シグネチャが同じであれば、どの実装でも交換可能
4. **Composability**: パイプライン構造を再利用しつつ、実装を変更できる
5. **Testability**: 各stepで異なる実装をテストしやすい
6. **Discoverability**: Auditor結果から利用可能な関数を自動発見
7. **Configuration-Driven**: コード変更なしで実験できる
8. **Self-Documenting**: バリデーションエラーが適切な Transform を提案
9. **Auto-Generation**: Skeleton定義からサンプル設定を自動生成、手動作業を完全排除

## Example Use Cases

### 0. Configuration Auto-Generation Workflow

```bash
# Step 1: Define skeleton (developer work - ~100 lines)
# apps/algo-trade/algo_trade_dag/skeleton.py

# Step 2: Auto-generate sample config (no manual work!)
uv run python -m xform_core.dag apps/algo-trade generate-config \
  --skeleton phase1_skeleton \
  --output configs/samples/phase1_sample.yaml \
  --show-alternatives

# Step 3: Validate generated config
uv run python -m xform_core.dag apps/algo-trade validate \
  configs/samples/phase1_sample.yaml

# Step 4: Run with generated config
uv run python -m xform_core.dag apps/algo-trade run \
  configs/samples/phase1_sample.yaml

# Step 5: User creates customized config based on sample
cp configs/samples/phase1_sample.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml to customize parameters
```

### 1. A/B Testing Different Feature Engineering Methods

```yaml
# Experiment A: RSI with period 14
phase2:
  steps:
    calculate_rsi:
      transform: "algo_trade_transforms.transforms.calculate_rsi"
      params:
        period: 14

# Experiment B: Custom RSI implementation
phase2:
  steps:
    calculate_rsi:
      transform: "algo_trade_transforms.experimental.calculate_rsi_optimized"
      params:
        period: 14
        optimization: "numba"
```

### 2. Swapping Data Providers

```yaml
# Production: Yahoo Finance
phase1:
  steps:
    fetch_data:
      transform: "algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv"

# Testing: Mock Data
phase1:
  steps:
    fetch_data:
      transform: "algo_trade_transforms.testing.fetch_mock_ohlcv"
```

### 3. Multiple CV Strategies

```yaml
# Strategy 1: Time Series Split
phase3:
  steps:
    cv_split:
      transform: "algo_trade_transforms.training.get_cv_splits_timeseries"
      params:
        n_splits: 5

# Strategy 2: Expanding Window
phase3:
  steps:
    cv_split:
      transform: "algo_trade_transforms.training.get_cv_splits_expanding"
      params:
        initial_train_size: 1000
```

### 4. Configuration Error Detection (Validation Example)

```yaml
# ❌ Invalid Configuration
phase2:
  steps:
    calculate_rsi:
      transform: "algo_trade_transforms.transforms.calculate_rsi_typo"  # Typo in function name
      params:
        period: 14
        unknown_param: "value"  # Invalid parameter

# Validation Output:
# ✗ Configuration validation failed:
#
#   [ERROR] phase2.calculate_rsi: TRANSFORM_NOT_FOUND
#     Transform 'algo_trade_transforms.transforms.calculate_rsi_typo' not found in registry
#     Suggestion: Available transforms:
#       - algo_trade_transforms.transforms.calculate_rsi
#
#   [ERROR] phase2.calculate_rsi: UNKNOWN_PARAMETER
#     Parameter 'unknown_param' not accepted by algo_trade_transforms.transforms.calculate_rsi
#     Suggestion: Valid parameters: period, symbol

# ✓ Fixed Configuration
phase2:
  steps:
    calculate_rsi:
      transform: "algo_trade_transforms.transforms.calculate_rsi"  # Fixed function name
      params:
        period: 14  # Removed invalid parameter
```

### 5. Type Mismatch Detection

```yaml
# ❌ Type Signature Mismatch
phase2:
  steps:
    feature_extraction:
      # This function expects pd.DataFrame but skeleton expects MultiAssetOHLCVFrame
      transform: "some_package.extract_features_from_dataframe"
      params:
        n_features: 10

# Validation Output:
# ✗ Configuration validation failed:
#
#   [ERROR] phase2.feature_extraction: TYPE_SIGNATURE_MISMATCH
#     Transform signature mismatch:
#       Expected: (MultiAssetOHLCVFrame,) -> FeatureFrame
#       Actual: (pd.DataFrame,) -> pd.DataFrame
#     Suggestion: Available transforms:
#       - algo_trade_transforms.transforms.select_features
#       - algo_trade_transforms.transforms.extract_custom_features

# ✓ Fixed Configuration
phase2:
  steps:
    feature_extraction:
      # Use transform with correct type signature
      transform: "algo_trade_transforms.transforms.select_features"
      params:
        feature_specs: [["USDJPY", "rsi", 14]]
```

## Summary

### Key Design Decisions

1. **Validation-First Approach**: 設定エラーは実行前に完全検出し、Fail Fast する
2. **Type-Driven Architecture**: 型シグネチャでパイプライン構造を定義し、実装を動的選択
3. **Configuration as Code**: YAML設定でTransform関数とパラメータを完全制御
4. **Self-Documenting System**: バリデーションエラーが適切な修正方法を提案
5. **Registry-Based Discovery**: Auditor が収集した Transform を型ベースで検索可能
6. **Core-App Separation**: DAG実行基盤はCore、Skeleton定義はApp固有で分離し最大限再利用
7. **Unified CLI Interface**: xform_auditor と同じパターンで、全アプリが同一インターフェースを使用
8. **Configuration Auto-Generation**: Skeleton定義から設定ファイルを自動生成、開発者の手動作業を排除

### Implementation Workflow

```
1. [Core] Auditor runs and populates Transform Registry
   ↓
2. [App] Define Skeleton (Phase1-4 type transformation flow)
   ↓
3. [User] Write pipeline configuration (YAML)
   ↓
4. [Core] Configuration Validator checks:
   - Transform existence
   - Type compatibility
   - Parameter completeness
   ↓
5. If validation fails → [Core] Show errors with suggestions
   If validation succeeds → Continue to execution
   ↓
6. [Core] DAG Executor resolves and executes transforms
```

### Core vs App Responsibilities

| Component | Core (xform-core) | App (algo-trade) | 実装コード量 |
|-----------|-------------------|------------------|------------|
| **Skeleton** | Base classes (PipelineStep, PipelineSkeleton) | Concrete definitions (phase1_skeleton, etc.) | Core: 50行, App: 100行 |
| **Validator** | Generic validation logic | - | Core: 200行 |
| **Resolver** | FQN→Function resolution | - | Core: 50行 |
| **Executor** | Generic execution engine | - | Core: 150行 |
| **Config** | Generic YAML loading | - | Core: 100行 |
| **CLI** | **Unified entry point with app auto-detection** | - | **Core: 150行, App: 0行** |
| **Transforms** | - | All @transform implementations | App: 既存実装 |
| **Types** | - | All TypedDict definitions | App: 既存定義 |

### Validation Guarantees

設定ファイルがバリデーションを通過すれば、以下が保証される：

✓ 全ての Transform FQN が Registry に存在  
✓ 全ての型シグネチャが Skeleton と互換  
✓ 全ての必須パラメータが提供済み  
✓ 全てのパラメータ名が有効  
✓ 全てのパラメータ型が一致  

→ **実行時エラーを大幅に削減**

### Related Documentation

- **Transform Guidelines**: [`/CLAUDE.md`](../../CLAUDE.md) - Transformer Function Guidelines
- **Architecture**: [`/doc/ARCHITECTURE.md`](../ARCHITECTURE.md)
- **Auto Annotation**: [`/doc/AUTO_ANNOTATION_RESOLUTION.md`](../AUTO_ANNOTATION_RESOLUTION.md)
- **Migration Guide**: [`/doc/MIGRATION_TO_DYNAMIC_DAG.md`](../MIGRATION_TO_DYNAMIC_DAG.md) - 手続き型DAGからの移行ガイド（詳細な比較含む）

