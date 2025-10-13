自動注釈解決設計書

1. 目的
- 入出力型だけを記述した通常の関数定義から、ExampleValue / Check を自動補完して TransformFn を生成する。
- TR001-TR009 の整合性を維持しつつ、関数実装者が Annotated メタデータを明示しなくても済むようにする。
- 実行時と静的解析時に同一の解決結果を得られるよう、共通のレジストリと解決手順を提供する。

2. 適用範囲
- xform-core の @transform デコレータ、および付随する TransformFn 正規化処理。
- xform-core が提供する mypy プラグイン。
- proj-dtypes で定義される型および Example/Check の既定値。
- apps/* 内で定義される全ての @transform 関数。

3. 要件
3.1 機能要件
- R1: 関数定義に通常の型注釈だけが記述されている場合でも、入力型をキーにして ExampleValue を補完できること。
- R2: 戻り値が通常の型注釈だけでも、戻り値型をキーにして Check を補完できること。
- R3: 補完された Example/Check は TransformFn 記録および監査 CLI から利用できること。
- R4: 設定された Example/Check をユーザーが明示的に指定した場合は、そちらを優先する。
- R5: 同一型に複数の Example/Check が紐づくケースでは、型→List 形式の単純なマッピングから順番に全てを適用できること（MVP要件）。
- R6: タグ・優先度などの追加メタデータは将来的な拡張とし、必要になった時に opt-in で利用できること。

3.2 非機能要件
- N1: 解決処理は依存方向（core → dtypes → apps）を崩さないこと。
- N2: mypy プラグインはレジストリ情報を参照可能な形で提供されること（循環禁止）。
- N3: 監査 CLI の Example/Check 解決ロジックと整合性が取れていること。
- N4: 実行時例外が利用者の注釈不足に起因する場合、明確なエラーを返すこと。

4. 全体アーキテクチャ
- xform-core
  - registry.py（新設）: 型ごとの ExampleResolver / CheckResolver を登録・参照する。
  - annotations.py: 型注釈からレジストリを利用して Example/Check を補完するヘルパーを実装。
  - transform.py: 通常の型注釈から TransformFn 正規化時に Annotated メタデータを生成して埋め込む。
  - plugin/: mypy プラグインで registry を参照できる API を用意し、TR003-TR006 を自動補完前提で評価。
- proj-dtypes
  - registry_setup.py（新設）: HLOCVSpec、FeatureMap、pd.DataFrame 等に対応する Example/Check を登録。
  - __init__.py: モジュール import 時に registry_setup.register_defaults() を呼ぶ。
- apps/
  - 既存の関数から冗長な Example/Check を段階的に撤去し、補完が動作することを確認する。

5. レジストリ仕様
5.1 データ構造
```
# MVP（必須）
ExampleRegistry = dict[str, list[ExampleValue[Any]]]
CheckRegistry = dict[str, list[CheckLiteral]]

# 拡張（オプション）
ExampleResolver = Protocol[Callable[[str, "ResolutionContext"], Sequence[ExampleValue[Any]]]]
CheckResolver = Protocol[Callable[[str, "ResolutionContext"], Sequence[CheckLiteral]]]

@dataclass
class ResolverEntry:
    key: ExampleKey           # 型の FQN を基本キーとする
    resolver: ExampleResolver | CheckResolver
    tags: frozenset[str]
    priority: int = 100       # 小さい値ほど優先
    compose: bool = True      # True の場合は同一 priority 帯で多重適用
    description: str = ""

ExampleKey = str  # 型の FQN（例: "pandas.core.frame.DataFrame"）

@dataclass
class ResolutionContext:
    slot: Literal["input", "output"]
    param_name: str | None
    extra_tags: frozenset[str]
```
- 最低限は単純な `dict[str, list[Any]]` で解決し、順序定義された複数 Example / Check を付与する。
- 現行 MVP では compose や優先度を考慮せず、マッチした要素を全て適用する実装とする。
- 拡張が必要になった場合は `ResolverEntry` を使う実装へ段階的に移行できるよう、同一ファイルで両者を記述する。
- tags / priority / compose はオプションの拡張要素として定義し、導入しない場合は無視される。

5.2 登録 API（案）
（MVP）
```
def register_example(key: str, example: ExampleValue[Any]) -> None: ...
def register_check(key: str, check: CheckLiteral) -> None: ...

def resolve_examples(key: str) -> Sequence[ExampleValue[Any]]: ...
def resolve_checks(key: str) -> Sequence[CheckLiteral]: ...
```
- Python 辞書を介して単純に append / read する実装で足りる。
- 解決処理は順序を変えずに複数要素を返す。

（拡張機能を導入する場合）
```
def register_example(entry: ResolverEntry) -> None: ...
def register_check(entry: ResolverEntry) -> None: ...

def resolve_examples(key: ExampleKey, *, context: ResolutionContext) -> Sequence[ExampleValue[Any]]: ...
def resolve_checks(key: ExampleKey, *, context: ResolutionContext) -> Sequence[CheckLiteral]: ...
```
- register 時に priority / tags / compose を指定できるようにし、後段の解決ポリシーを柔軟にする。
- `resolve_*` は複数の Example/Check を返し、TransformFn には全てを Annotated メタとして埋め込む。
- 代表的なキー: `"proj_dtypes.hlocv_spec.HLOCVSpec"`、`"pandas.core.frame.DataFrame"`。

5.3 MVP: 単純マッピング
- 型の FQN をキーにし、`ExampleValue` / `CheckLiteral` のリストを格納する `dict[str, list[Any]]` を基本形とする。
- 解決時は登録順を尊重し、全ての Example / Check を Annotated メタデータに追加する。
- 追加メタ情報（タグや priority）が無い前提のため、複合チェックはリストの複数要素としてそのまま適用する。
- 曖昧さ判定は「同じ型に登録が無い」場合のみ Missing とし、ある限りは順番通り適用する。
- プロジェクトごとの登録は `registry_setup` など単一モジュールで一括定義する。

5.4 拡張機能（オプション）
- タグ / 優先度 / compose フラグは、型だけでは識別できない用途が生じた場合に導入する拡張要素とする。
- 拡張を利用する場合は `ResolverEntry` と `ResolutionContext` を opt-in で使用し、既存の `dict[str, list[Any]]` 形式からの互換ラッパーを提供する。
- タグ命名は `snake_case`、`用途_データ種別` を推奨。priority は 10/50/100 など数段階を基本値とする。
- 拡張機能を有効化した場合でも、MVP の単純マッピング経由での登録を崩さない（未指定の項目はデフォルト値で補う）。

5.5 proj-dtypes における登録方針
- MVP では `proj_dtypes.registry_setup.register_defaults()` 内で型→List の辞書に対して Example / Check を append するだけでよい。
- HLOCVSpec には単一 Example（乱数 seed は 42 など固定値）を、`pd.DataFrame` には複数チェック（行数 > 0 / Null 検証など）を登録する。
- 拡張機能を導入するタイミングでは、既存の辞書定義を `ResolverEntry` にラップするユーティリティを用意し、段階的にタグ・優先度を設定する。
- apps 側で更に厳しい制約を掛けたい場合は、追加の Example / Check を辞書に append し、実行順（リスト順）で意図を表現する。
- CI では registry_setup が呼ばれているか、辞書の空エントリが無いかを検査し、登録漏れを検出する。

5.6 異常系データ型と検査のサンプル
- DataFrame の部分集合を扱う場合は、`TypedDict` や `Protocol` を用いて列スキーマを定義し、その型名をレジストリキーにする。
  ```python
  class PriceBarsFrame(Protocol):
      close: pd.Series
      volume: pd.Series

  register_check(
      "proj_dtypes.frames.PriceBarsFrame",
      Check("proj_dtypes.checks.check_price_bars_columns"),
  )
  ```
- 文字列カテゴリの制限は、`enum.StrEnum` や `Literal` ベースの独自型で表現し、該当型のチェックで禁止カテゴリを検知する。
  ```python
  class MarketRegime(StrEnum):
      BULL = "bull"
      BEAR = "bear"
      SIDEWAYS = "sideways"

  register_check(
      "proj_dtypes.types.MarketRegime",
      Check("proj_dtypes.checks.check_market_regime_known"),
  )
  ```
- 異常サンプルは ExampleValue 側で生成し、チェックが例外を送出することで異常ケースを検知できるようにする（例: 欠損列を含む DataFrame、未定義カテゴリを含む文字列）。
- これらの独自型を利用する Transform では通常の型注釈（`bars: PriceBarsFrame` や `regime: MarketRegime`）を書くことで、自動補完されたチェックが異常検知を担保する。

6. 解決フロー
6.1 入力 Example の補完
- ステップ1: 関数引数の型ヒントを取得し、標準化した ExampleKey を生成。
- ステップ2: 関数定義に ExampleValue が明示されていればそれを採用し、無い場合は registry.resolve_examples(key) を呼び出す。
- ステップ3: 返却された ExampleValue 群を `Annotated[param_type, *examples]` として組み立て、TransformFn レコードに格納。
- ステップ4: 解決に失敗した場合は `MissingExampleError` を送出。

6.2 出力 Check の補完
- ステップ1: 戻り値型ヒントを解析し、ExampleKey を生成。
- ステップ2: 戻り値注釈に Check が明示されていればそれを採用し、無い場合は registry.resolve_checks(key) を呼ぶ。
- ステップ3: 返却された Check 群を `Annotated[return_type, *checks]` として組み立て、TransformFn レコードに格納。
- ステップ4: 返却数が 0 件の場合のみ `MissingCheckError` を送出する（複数件は順番通り全て適用）。

7. mypy プラグイン更新
- TR003/TR004: 関数定義に Annotated が無くても、registry.resolve_examples(key) が非空なら要件を満たすと判断する。
- TR005/TR006: 戻り値に Annotated が無くても、registry.resolve_checks(key) が非空なら合格とする。
- TR007/TR008: 自動補完された Check を Literal[str] として type-checker 内で扱えるよう、プラグインが内部で Annotated を構築する。
- TR001/TR002: 最初の引数が Annotated でなくても、`Annotated[type, ...]` に変換する前提で検証できるようエラーメッセージを更新する。
- 実装メモ: mypy プラグインは transform 解析時にレジストリ API を呼び、補完が成功すれば TR00x のエラーを発生させない。補完が失敗した場合のみ既存のエラーを出す。

8. 追加実装タスク
- T1: registry モジュール（辞書ベース実装）の整備と単体テスト。登録順で全要素を返すことを確認する。
- T2: @transform の正規化処理更新。通常の型注釈から Annotated を組み立てる実装とテストを追加する。
- T3: mypy プラグインの拡張。registry を参照して TR001-TR009 を再評価するテストを構築する。
- T4: proj-dtypes の既定登録処理（seed=42 での Example、複数 Check）と回帰テスト。
- T5: apps の注釈削減対応。`generate_price_bars` / `compute_feature_map` 等を素の型注釈に置き換え、監査 CLI/pytest を再実行する。
- T6: 異常系データ型（列制約付き DataFrame、限定カテゴリ文字列など）のチェック関数と Example を実装し、回帰テストを追加する。

9. ロールアウト計画
- Phase1: registry 実装、単体テスト、CI 反映。
- Phase2: mypy プラグイン対応、型チェック CI の確認。
- Phase3: proj-dtypes での既定登録、`generate_price_bars` 等を移行。
- Phase4: 監査 CLI が自動補完に対応することを確認（将来実装時）。
- Phase5: ドキュメント更新、運用ガイドライン整備。

10. リスクと対応
- RISK1: 登録漏れによる MissingExampleError 多発 → CI に登録チェックを追加。
- RISK2: DataFrame 等汎用型での誤マッチ → タグ必須化と型定義（TypedDict/Protocol）で明確化。
- RISK3: mypy プラグインと実行時解決の不整合 → registry を単一ソースとし、両者で同じ API を利用。
- RISK4: 起動順依存（登録前に解決が走る） → proj-dtypes の __all__ 読み込み時に register_defaults を確実に呼ぶ。

11. 未決定事項 / オープンクエスチョン
- Q1: ExampleResolver のキャッシュ戦略（seed や乱数状態の扱い）。
- Q2: Check 解決で複数候補を compose する際の既定優先順位（CI/設定ファイルで制御すべきか）。
- Q3: 監査 CLI 実装時の遅延 import による登録順序の影響。
- Q4: mypy プラグインでのタグ指定の表現（Literal[str] での記述可否）。

12. 実装スケッチ
```
# packages/xform-core/xform_core/registry.py（MVP）

_example_registry: dict[str, list[ExampleValue[Any]]] = defaultdict(list)
_check_registry: dict[str, list[CheckLiteral]] = defaultdict(list)

def register_example(key: str, example: ExampleValue[Any]) -> None:
    _example_registry[key].append(example)

def register_check(key: str, check: CheckLiteral) -> None:
    _check_registry[key].append(check)

def resolve_examples(key: str) -> Sequence[ExampleValue[Any]]:
    return list(_example_registry.get(key, []))

def resolve_checks(key: str) -> Sequence[CheckLiteral]:
    return list(_check_registry.get(key, []))

# packages/xform-core/xform_core/transform.py（抜粋）

def _auto_annotate_parameter(param: inspect.Parameter) -> inspect.Parameter:
    key = qualname_from_annotation(param.annotation)
    examples = resolve_examples(key)
    if not examples:
        raise MissingExampleError(param.name, key)
    return param.replace(annotation=Annotated[param.annotation, *examples])

def _auto_annotate_return(annotation: Any) -> Any:
    key = qualname_from_annotation(annotation)
    checks = resolve_checks(key)
    if not checks:
        raise MissingCheckError(key)
    return Annotated[annotation, *checks]

# packages/proj-dtypes/proj_dtypes/registry_setup.py（MVP）

def register_defaults() -> None:
    register_example("proj_dtypes.hlocv_spec.HLOCVSpec", ExampleValue(HLOCVSpec(n=16, seed=42)))
    register_example(
        "pandas.core.frame.DataFrame",
        ExampleValue(gen_hlocv(HLOCVSpec(n=16, seed=42))),
    )
    register_check("pandas.core.frame.DataFrame", Check("proj_dtypes.checks.check_hlocv_dataframe_length"))
    register_check("pandas.core.frame.DataFrame", Check("proj_dtypes.checks.check_hlocv_dataframe_notnull"))
    register_check("proj_dtypes.types.FeatureMap", Check("proj_dtypes.checks.check_feature_map"))

# apps/pipeline-app/pipeline_app/__init__.py

from proj_dtypes import registry_setup
registry_setup.register_defaults()

@transform
def generate_price_bars(...) -> pd.DataFrame:
    ...  # Example / Check は自動補完

# mypy プラグイン（概略）

def analyze_transform_function(ctx: FunctionContext) -> None:
    param_type = ctx.arg_types[0][0]
    examples = registry.resolve_examples(qualname_from_mypy_type(param_type))
    if not examples:
        ctx.api.fail("TR003: ExampleValue が見つかりません", ctx.context)
    return_type = ctx.default_return_type
    checks = registry.resolve_checks(qualname_from_mypy_type(return_type))
    if not checks:
        ctx.api.fail("TR005: Check が見つかりません", ctx.context)

# 拡張機能（タグ/優先度）を使う場合のラッパー例

def register_example_entry(entry: ResolverEntry) -> None:
    ctx = ResolutionContext(slot="input", param_name=None, extra_tags=frozenset())
    register_example(entry.key, entry.resolver(entry.key, ctx)[0])
    # ResolverEntry を単純マップへ落とし込む等の互換層を用意
```
- MVP では字義通りの辞書マッピングで完結させ、運用面の複雑さを回避する。
- 拡張機能を有効化する場合は `ResolverEntry` を単純マップへ落とし込むアダプタ／または別 API を提供し、段階的移行を可能にする。
- registry_setup は idempotent かつ多重呼び出し時に副作用が無いようにし、apps の import シーケンスに依存しない設計とする。

13. エラーハンドリングと診断
- 例外は `ResolutionError` を基底クラスとし、`MissingExampleError` / `MissingCheckError` / `RegistryNotInitializedError` を派生させる。全て `xform_core.exceptions` に集約する。
- 各例外には `param_name` / `example_key` / `available_keys` 等のコンテキストを保持させ、`str(error)` で人間が読めるメッセージ（例: `"generate_price_bars.X: ExampleValue for pandas.core.frame.DataFrame が未登録です"`）を返す。
- 解決処理は例外送出前に `registry.list_registered_keys(slot="input")` のような診断 API を参照できるようにし、CLI や mypy プラグインが候補一覧を提示できるようにする。
- `@transform` デコレータは補完フェーズで例外を捕捉し、`TransformFnErrorReport`（仮称）に変換して呼び出し元へ伝搬させる。CLI はこのレポートを集約し、関数単位で失敗理由を出力する。
- mypy プラグインは同じ例外情報を `ctx.api.fail(...)` に流用し、実行時と静的解析で同一メッセージ・同一キーを報告する。

14. 初期化フローと依存順序
- `xform_core.registry` は import 時点では空の辞書だけを持つ。`proj_dtypes.registry_setup.register_defaults()` が呼ばれた後にキーが埋まる前提とする。
- `proj_dtypes/__init__.py` で `register_defaults()` を即時実行しつつ、CLI やテストが明示的に再実行しても多重登録にならないように `if key in registry:` チェックを入れる。
- apps 側では `from proj_dtypes import registry_setup` を最上位で import し、副作用として登録が完了することを保証する。移行期間は `apps/__init__.py` でフェールファストチェック（`assert registry.is_initialized()`）を入れて検出する。
- mypy プラグインからは import ルールの制約上レジストリ初期化コードを直接読めないため、`pyproject.toml` 側で `mypy_path` に `packages/proj-dtypes/src` を含め、`registry_snapshot.json`（make check 時に生成）を参照する案も検討する。MVP ではプラグイン実行時に標準 Python import を試み、失敗時は明示的なワーニングを表示する。
- テストでは `monkeypatch` で `_example_registry.clear()` を行った後、`register_defaults()` を呼び直すヘルパーを用意し、初期化順依存のバグを再現できるようにする。

15. テスト戦略
- `packages/xform-core/tests/test_registry.py`: 登録順保持、重複登録防止、未登録時の例外メッセージを検証する。
- `packages/xform-core/tests/test_transform_auto_annotation.py`: 明示的な Annotated が無い関数に対して Example/Check が補完されること、`MissingExampleError` が正しく送出される異常系を確認する。
- `packages/xform-core/tests/test_plugin_resolution.py`: mypy プラグイン用の仮想モジュールを生成し、registry 解決結果に応じて TR003/TR005 のエラーが抑制されることを確認する（`pytest-mypy-plugins` の利用を想定）。
- `packages/proj-dtypes/tests/test_registry_setup.py`: `register_defaults()` 後の辞書内容をアサートし、DataFrame 生成が安定しているか（seed）をチェックする。
- 結合テストとして `uv run pytest apps/pipeline-app/tests/test_generate_price_bars.py` を追加し、TransformFn 正規化→自動補完→関数実行→チェック実施までの一連の流れを検証する。
- 将来的には `xform-auditor` の CLI テストで `registry` の状態をモックし、欠損例外が CLI 出力に反映されることを回帰テストする。

16. 明示 Annotated との互換方針
- 関数定義で既に `Annotated` が指定されている場合、`ExampleValue` / `Check` の両方とも「ユーザー優先」とし、レジストリからの補完はスキップする。
- `ExampleType` が指定されている場合は、解決結果として `ExampleValue` を併用できるようにし、`Annotated[T, ExampleType[SampleSpec], ExampleValue[generated]]` の形を許容する。順序は元の注釈を保ち、追記は末尾に行う。
- 明示注釈と自動補完の重複が生じた場合は、`dedupe_metadata()` ヘルパーで `CheckLiteral` / `ExampleValue` の値を比較し、重複を一つにまとめる。
- `@transform` デコレータには `auto_annotation=False` のようなフラグを用意し、互換性リスクがある関数では明示的に自動補完を無効化できるようにする。
- 将来的にタグベース解決を導入した際は、`Annotated[T, AutoTag["foo"]]` のような軽量メタを追加することで、ユーザーが解決グループを選択できる設計を想定する。

17. 運用ガイドラインと導入手順
- `make setup` 実行後に `uv run python -m proj_dtypes.registry_setup --check`（簡易スクリプト）で登録状況を検査し、CI でも同じチェックを走らせる。
- 新しい型を追加する際は、(1) 型定義、(2) ExampleValue 生成、(3) Check 関数（最低 1 件）、(4) registry への登録、(5) テスト追加、の順序で作業するチェックリストを `doc/DEVELOPER_GUIDE.md` に追記する。
- プロジェクト外部で TransformFn を利用する場合は、`registry.export_snapshot(path)` で JSON 出力し、別プロセスで `import_snapshot` することで依存循環を避ける。
- 既存関数の移行は、まず戻り値の Check をレジストリへ移行し、次に入力 Example を移行、最後に Annotated を削除する段階的アプローチを推奨する。各段階で `make typecheck` / `uv run pytest` を回して回帰を防ぐ。
- ドキュメント類（本設計書含む）は実装完了後にバージョンタグへリンクし、仕様の逸脱が発生した場合は本ドキュメントを更新する運用ポリシーを設定する。
