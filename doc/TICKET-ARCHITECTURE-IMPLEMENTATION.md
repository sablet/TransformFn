# doc/ARCHITECTURE.md 実装タスクチケット

## 背景
- `doc/ARCHITECTURE.md` に TransformFn 基盤の仕様と、各 `packages/*` に対するテスト追加方針が追記された。
- 仕様に沿ったモノレポ構成・コアライブラリ・監査 CLI は未実装であり、特に `xform-core` の MVP とその正常系/異常系テストを最優先で整備する必要がある。
- パッケージ管理は `uv` で行う。

## 重要な実装方針
**CRITICAL**: このチケットの作業項目を実装する際は、以下の方針を厳守すること：

1. **途中で止まらない**: チケットの全項目が完了するまで、ユーザーへの中間報告や確認を求めずに作業を継続する。
2. **品質チェックの完全通過**: 各チェックリスト項目の実装完了後、必ず `make check` を実行し、全ての品質チェック（duplication, format, lint, typecheck, complexity）が通るようにリファクタリングを完了する。
3. **完了報告のタイミング**: 全てのチェックリスト項目が完了し、かつ `make check` が成功した時点で、初めてユーザーに完了報告を行う。

## ゴール
- `doc/ARCHITECTURE.md` のアーキテクチャを最小実装し、`xform-core` を中心とした TransformFn 基盤を動作させる。
- `xform-core` の MVP を実装し、正常系と異常系テストを含む包括的なテストスイートを追加する。
- `packages/*` の各パッケージに対応するテストを追加し、`uv` 管理下で一貫して実行できるようにする。
- CI で mypy＋監査 CLI（必要に応じ pytest）を実行し、失敗時に検知できる状態にする。

## 成果物
- `packages/xform-core/`：TransformFn 正規化ロジックとメタ型、mypy プラグイン、`tests/` ディレクトリ。
- `packages/proj-dtypes/`：ドメイン型・Example/Check 実装と `tests/` を追加。
- `packages/xform-auditor/`：監査 CLI 実装と `tests/` を追加。
- `apps/pipeline-app/`：Transform 関数サンプル、DAG、実行例、`tests/` を追加。
- `Makefile` / `pyproject.toml` / `uv.lock`：`uv` ベースのセットアップ・テストコマンドを整備。
- `doc/`：利用手順・開発者向けガイドの更新。

## 作業項目（チェックリスト）
### 1. モノレポ構成とビルド基盤
- [ ] `packages/`・`apps/` ディレクトリ構成を整備し、`pyproject.toml` を PEP 517 & `uv` 管理に合わせて更新する。
- [ ] 各パッケージに `pyproject.toml` / `__init__.py` / `py.typed` を配置し、依存関係・開発用 extras を定義する。
- [ ] `Makefile` に `uv` を利用した `setup`, `lint`, `typecheck`, `audit`, `test` ターゲットを追加し、ローカル開発フローを文書化する。

### 2. `xform-core` MVP（最優先）
- [ ] `@transform` デコレータと TransformFn データモデル（UUID・Schema 抽出・CodeRef ハッシュ）を実装する。
- [ ] TypedDict / Dataclass から入出力 Schema・ParamSchema を抽出するヘルパーを実装する。
- [ ] メタ型（`ExampleType`, `ExampleValue`, `Check`）と Annotated 解析ユーティリティを実装する（`@transform` 関数シグネチャから Example/Check メタデータを抽出し TransformFn 正規化に渡す役割）。
- [ ] `xform_core.dtype_rules.plugin` の mypy プラグインで TR001〜TR009 をカバーし、段階的に有効化する。
- [ ] `packages/xform-core/tests/` に正常系と異常系（各 TR00x）を網羅するテストを作成し、`uv run` で再現できるようにする。

### 3. Example/Check ユーティリティとリポジトリ横断の登録
- [ ] Example/Check メタ情報の登録 API を実装し、診断メッセージ・検証ロジックを整備する（`xform-core` が抽出したメタデータをパッケージ横断で共有・参照できるレジストリ）。
- [ ] 将来拡張点（ExampleList, CheckPair など）をスタブ or コメントで記載し、テストで基本的な登録/取得ケースを検証する。

### 4. `proj-dtypes` の実装とテスト
- [ ] HLOCVSpec などのドメイン型を定義し、Example データ生成（`gen_hlocv` など）と Check 関数を実装する。
- [ ] `packages/proj-dtypes/tests/` に Example 生成・Check 評価の正常/異常ケースを追加する。

### 5. `xform-auditor` CLI
- [ ] 対象モジュール探索と `@transform` 関数列挙ロジックを実装する。
- [ ] Example/Check から入力生成→関数実行→結果検証→集計までのコア処理を実装する。
- [ ] CLI エントリポイント、テキスト/JSON 出力、終了コード制御を整備し、`packages/xform-auditor/tests/` に正常/異常ケースを追加する。

### 6. サンプルアプリ (`apps/pipeline-app`)
- [ ] Transform 関数群と DAG を実装し、`doc/ARCHITECTURE.md` の仕様をデモできるようにする。
- [ ] 簡易ランナー（キャッシュキー計算、Artifact 記録スタブ）を実装し、監査 CLI 実行例を `uv run` で再現できるようにする。
- [ ] `apps/pipeline-app/tests/` に変換関数の smoke テストと DAG 実行テストを追加する。

### 7. キャッシュ・アーティファクト処理
- [ ] CacheKey 生成とハッシュ計算（transform_fn_id, params, input_hashes, code_hash, env_hash）を実装する。
- [ ] Artifact / Provenance データクラスと永続化インタフェース（ファイルシステムの最小実装）を用意する。
- [ ] テストで CacheKey/Artifact の生成・比較・保存を検証する。

### Appendix. CI / 品質ゲート・ドキュメント
- [ ] CI（例：GitHub Actions）で `uv run mypy` と `uv run python -m xform_auditor ...`（必要なら `uv run pytest`）を実行し、失敗時に検知できるよう設定する。
- [ ] `doc/DEVELOPER_GUIDE.md`（新規）や既存ドキュメントに `uv` ベースのセットアップ／テスト手順を追記する。
- [ ] 開発・運用中の既知の制約/TODO をドキュメントまたはコードコメントに明記する。

## リスク・検討事項
- mypy プラグインの実装コストが高いため、チェックを段階的に追加しつつテストで検証する。
- `xform-core` テストの異常系ケースは新規仕様に合わせて網羅的に準備する必要があり、優先的にアサインする。
- Pandas など Example 生成に必要な依存のサイズ・CI 時間に注意する。
- Python 3.11 を前提とし、他バージョン対応は後回しにする。

## 完了条件
- `make typecheck`（`uv run mypy`）と `make audit`（`uv run python -m xform_auditor ...`）が成功する。
- `xform-core` 含む各パッケージで正常系/異常系テストが `uv run pytest` 等で成功する。
- サンプル TransformFn に対して監査 CLI を実行すると、Example 生成→Check 検証まで成功する。
- オンボーディング手順が `doc/` に記載され、`uv` を用いて 30 分以内に着手できる。
- 重要な TODO / 未実装箇所がドキュメントまたはコード内で追跡されている。

## 関連資料
- `doc/ARCHITECTURE.md`
- 将来作成予定：`doc/DEVELOPER_GUIDE.md`, `doc/AUDITOR_USAGE.md`
