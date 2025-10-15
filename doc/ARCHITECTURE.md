設計・仕様書（サマリ）

本書は、普通の Python 関数を 型＋メタ情報から正規化して データ変換ノード（TransformFn） として扱い、
DAG で配線し、完全キーによるキャッシュと 静的検査（mypy）・**簡易実行監査（CLI）**で品質を担保するための設計・仕様をまとめたものです。
実装は Python 3.11 以降を前提とします。

⸻

1. 目的 / ゴール
	•	記述コスト最小で、変換関数・データ型・例・検査を一体化（型注釈に集約）
	•	関数の正規化：関数→TransformFn レコードへ収縮（入出力スキーマ、パラメータ、コード参照）
	•	再現可能性：キャッシュキーに 関数・入力・パラメータ・コード・環境を含める
	•	静的ガード：mypy プラグインで Annotated + Example/Check + docstring を強制
	•	テスト最小化：pytest を書かずとも Annotation の Example/Check だけで 簡易実行・集計
	•	層の分離：共通（メタ処理系）とプロジェクト固有（型・例・検査・関数）を分離

非ゴール：フル機能のワークフローエンジン、スケジューラ、分散実行。必要に応じて後付け可能。

⸻

2. アーキテクチャ（層と依存）

xform-core（共通） ──▶ proj-dtypes（プロジェクト固有の型/例/検査） ──▶ apps/*（@transform 関数群・DAG）
                         ▲
                         └── xform-auditor（共通：Annotation 監査CLI。apps から利用）

	•	xform-core：@transform、メタ型（ExampleType, ExampleValue, Check）、mypy プラグイン
	•	proj-dtypes：n 個の カスタムデータ型、それに紐づく 例 と チェック関数、生成器（HLOCVSpec 等）
	•	apps/：m 個の 変換関数 と DAG 宣言
	•	xform-auditor：Annotation のみを用いて 簡易テスト & レポートする共通 CLI

⸻

3. データモデル（正規化オブジェクト）

データ変換関数はスキーマ付きレコード TransformFn として扱います。

classDiagram
class TransformFn {
  +UUID id
  +string name
  +string version
  +Schema input_schema
  +Schema output_schema
  +ParamSchema param_schema
  +CodeRef code_ref
  +string engine   // "python" など
  +bool is_pure
}

class Schema { +string name; +Field[] fields; +string primary_key? }
class Field { +string name; +string dtype; +bool nullable; +string description? }
class ParamSchema { +ParamField[] params }
class ParamField { +name; +dtype; +default?; +required }
class CodeRef { +module_path; +git_commit?; +container_image?; +runtime_hash? }

class Node { +UUID id; +TransformFn fn; +ParamValue param_value; +string name }
class Edge { +UUID from_node; +UUID to_node; +string via_slot }
class DAG  { +UUID id; +Node[] nodes; +Edge[] edges }

class Artifact {
  +UUID id; +Schema schema; +string uri; +string format
  +ContentHash content_hash; +Provenance provenance
  +Metrics? metrics; +int? size_bytes; +string[] tags
}
class Provenance { +UUID produced_by_node; +CacheKey cache_key; +UUID[] inputs; +ts created_at }
class CacheKey {
  +string key; +UUID transform_fn_id; +ParamValue params
  +ContentHash[] input_hashes; +string code_hash; +string env_hash
}

完全キャッシュキー = ハッシュ(transform_fn_id, params, input_hashes[], code_hash, env_hash)

⸻

4. 関数→TransformFn への収縮（正規化）
	•	記述：普通の Python 関数に @transform を付け、**型注釈（Annotated）**で入出力を記述
	•	抽出：inspect + 型ヒントから Schema / ParamSchema を自動生成
	•	コード参照：CodeRef.module_path="pkg.mod:func" と code_hash=sha256(source) を記録

from xform_core.transforms_core import transform
from typing import Annotated, TypedDict

class In(TypedDict):  # 例
    text: str
class Out(TypedDict):
    tokens: list[str]

@transform
def tokenize(X: Annotated[In, ...], lower: bool = True) -> Annotated[Out, ...]:
    ...


⸻

5. 型注釈と Example / Check

メタ型（xform-core）

class ExampleType(Generic[T]): ...   # 「型が T である例を用意できる」宣言（静的）
class ExampleValue(Generic[T]): ...  # 「この値（or 仕様）は T に適合する」静的検査
class Check(Generic[Literal["pkg.func"]]): ...  # 実在する検査関数への FQN

典型パターン
	•	入力：X: Annotated[pd.DataFrame, ExampleValue[HLOCVSpec(n=1000)]]
→ 監査 CLI が HLOCVSpec から DataFrame を生成
	•	出力：-> Annotated[dict[str, float], Check["algo_trade_dtype.checks.check_feature_map"]]

⸻

6. HLOCV の例データ（仕様オブジェクト）

HLOCVSpec（`apps/algo-trade-app/algo_trade_dtype/` に配置）で 高値/安値/終値/出来高の制約を満たす DF を生成。
	•	GBM ベースの終値、open_t = close_{t-1}
	•	high ≥ max(open, close), low ≤ min(open, close) を常に満たす
	•	出来高は |return|・曜日と相関

監査 CLI は ExampleValue[HLOCVSpec(...)] を検出し、アプリ固有のジェネレータを通じて DF を作る。

⸻

7. 静的解析ルール（mypy プラグイン / xform-core）

@transform 関数に対して以下を 型チェック時に強制（エラーコード TR00x）：
	1.	TR001：第1引数が存在
	2.	TR002：入力の第1引数が Annotated[...]
	3.	TR003：入力 Annotated に ExampleType[...] または ExampleValue[...] が含まれる
	4.	TR004：ExampleValue[U] が入力基底型に適合（U <: base）
	5.	TR005：返り値が Annotated[...]
	6.	TR006：返り値 Annotated に Check["pkg.func"] が含まれる
	7.	TR007：Check[...] の引数が 文字列リテラル FQN
	8.	TR008：Check 参照が 実在する関数
	9.	TR009：docstring が存在

プラグインは apps 側の pyproject.toml で
plugins = ["xform_core.dtype_rules.plugin"] を指定して有効化。

⸻

8. 監査 CLI（xform-auditor）

目的：pytest を書かずに、Annotation の Example/Check だけで 簡易テストを一括実行し、結果を要約。
	•	入力：モジュール名またはディレクトリ（内部で @transform 関数を探索）
	•	動作：
	1.	入力 Annotated から Example を構築
	•	ExampleValue[HLOCVSpec] → gen_hlocv(spec)
	•	ExampleType[pd.DataFrame] → 最小 DF（拡張可能なフック）
	2.	関数を デフォルト引数で実行
	3.	返り値 Annotated の Check 関数を実行
	4.	OK / VIOLATION（検査失敗）/ ERROR（実行 or 検査中例外）/ MISSING（例が作れず未実行） を集計
	•	出力：標準出力のサマリ、必要に応じ JSON/HTML（任意拡張）
	•	終了コード：違反やエラーがあれば 1、それ以外 0

実行例

python -m xform_auditor pipeline_app.transforms
# or 指定ディレクトリ
python -m xform_auditor apps/pipeline-app/pipeline_app


⸻

9. リポジトリ構成（推奨：モノレポ）

repo/
├─ packages/
│  ├─ xform-core/
│  │  └─ xform_core/{meta.py, transforms_core.py, dtype_rules/plugin.py, type_metadata.py}
│  │  └─ tests/{unit, integration}
│  └─ xform-auditor/
│     └─ xform_auditor/{auditor.py, examples.py, discover.py, report.py(任意)}
│     └─ tests/{unit, integration}
└─ apps/
   ├─ algo-trade-app/
   │  ├─ algo_trade_dtype/{types.py, generators.py, checks.py, registry.py}
   │  └─ algo_trade_app/{transforms.py, dag.py}
   └─ pipeline-app/
      └─ pipeline_app/{transforms.py, dag.py, faulty_transforms.py}

アプリごとの dtype パッケージは `RegisteredType` で Example/Check を宣言的に登録する。依存は `xform-core` → `apps/*` の一方向を維持する。

⸻

10. CI / 品質ゲート
	1.	mypy（静的規約）
	•	apps でプラグイン有効化：plugins = ["xform_core.dtype_rules.plugin"]
	•	Annotated + Example/Check + docstring 未満は ビルド失敗
	2.	xform-auditor（簡易実行）
	•	python -m xform_auditor pipeline_app.transforms
	•	変換関数を Example で実走・Check で検証
	•	違反/エラーがあれば CI 失敗

（必要に応じ pytest/hypothesis を追加）

⸻

11. パイプライン記述とキャッシュ
	•	DAG：Node(fn=TransformFn, param_value=...) を Edge(via_slot=...) で接続
	•	実行ランナー（最小）：トポロジカル順に
	1.	入力アーティファクトの content_hash 収集
	2.	CacheKey 生成 → キャッシュヒットならスキップ
	3.	ミス時に関数を実行し、Artifact と Provenance を保存

実行ランナーは将来拡張項目。まずは 正規化・静的検査・簡易実行 の三点を固める。

⸻

12. 拡張ポイント
	•	Example の拡張：ExampleList[T]（複数例）/ ExampleFactory[Callable[[], T]]
	•	Check の拡張：入出力ペア不変条件 CheckPair["pkg.func"]
	•	JSON Schema 生成：Schema から外部ツール連携
	•	環境ハッシュ：sys.version + pip freeze から env_hash 作成
	•	import-linter：依存逆流の禁止（packages→apps の import を禁止）

⸻

13. 最小インタフェース集
	•	@transform(fn)：実行時は no-op、型検査と収縮トリガー用
	•	ExampleType[T] / ExampleValue[T] / Check["pkg.func"]：メタ型
	•	mypy プラグイン：xform_core.dtype_rules.plugin
	•	監査 CLI：python -m xform_auditor <module_or_path>

⸻

まとめ
	•	書き方の規約（Annotated + Example/Check + docstring）を mypy で強制
	•	最小テストは Annotation だけに依存し、**CLI（xform-auditor）**で自動実行
	•	関数の正規化・プロビナンス・キャッシュで再現可能性を担保
	•	層の分離で再利用とスケールに耐える

この設計で、n 個の型 × m 個の関数が増えても、
記述負荷は最小・品質は静的＆簡易実行で担保・再現性/監査性も確保できます。
