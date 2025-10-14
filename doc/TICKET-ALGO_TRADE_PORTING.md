# algo_trade_v3 移植タスクチケット

## 背景
- `doc/ALGO_TRADE_V3_PORTING.md` で、`algo_trade_v3` の 3 サブプロジェクトから TransformFn へ移植可能な純粋計算ロジックとデータ型を整理した。
- TransformFn 側では透過的なキャッシュ層と `@transform` 正規化基盤を整備する予定であり、既存コードのファイル IO / キャッシュ目的の処理を直接移植する必要はない。
- `train_predict_server/experiments/` を参考に、複数通貨（通貨ペア + 通貨インデックス）を入力として最も上昇期待値の高い通貨を BUY するシミュレーションまで評価できるパイプラインを構築したい。

## 重要な実装方針
**CRITICAL**: 以下の方針を厳守すること。

1. **純粋計算のみを移植対象にする**: ファイル IO・parquet キャッシュ・Selenium ダウンロードなどの副作用は TransformFn の `apps` 層へ持ち込まず、必要なら外部ユーティリティに切り出す。TransformFn には DataFrame/配列を受け取って結果を返す純粋計算ノードのみを実装する。
2. **データ型は TypedDict / Pydantic で再定義**: 既存の dataclass / Pydantic モデルを参考にしつつ、新しい `datatype` として TransformFn 用に最小限のフィールドへ整理する。ファイルパスではなくデータ ID や統計値を扱う。
3. **評価パイプラインを意識した分解**: 特徴量生成・ターゲット生成・モデル学習・シミュレーションを DAG 上で再利用できるよう、計算ステップを小さく分割する。`experiments/` の BUY シナリオを再現できることを常に意識する。

## ゴール
- `algo_trade_v3` の純粋計算ロジック（インデックス計算、テクニカル指標、学習・評価関数など）を TransformFn の `packages` / `apps` に再配置し、IO 依存を排除した形で再利用可能にする。
- 複数通貨・指標を入力に持つ学習 → 予測 → BUY シミュレーションの TransformFn DAG を試作し、監査 CLI と `audit` コマンドで検証できる状態を作る。
- 透過キャッシュを前提とした `datatype` / `transformer` / `Check` を整備し、今後の DAG 拡張に耐えられる設計をまとめる。

## 成果物
- `packages/` 配下: 移植した計算ロジックを収める新規 or 既存モジュール（例: `xform-core` 追加ユーティリティ、`proj-dtypes` の通貨関連型、`xform-auditor` の評価 Check など）。
- `apps/pipeline-app/` 配下: 複数通貨入力を想定した Transform 関数群と BUY シミュレーション用パイプライン、サンプル DAG、Example Data。
- `tests/`: 各移植モジュールに対応する正常系 / 異常系 / 境界テスト。
- `doc/`: 本チケット、移植済み API の利用ガイド、評価パイプラインの説明を追記。

## 作業項目（チェックリスト）
### 1. 設計と準備
- [ ] `doc/ALGO_TRADE_V3_PORTING.md` を確認し、移植対象と対象外（IO / Selenium / parquet 保存など）を最終決定する。
- [ ] 既存 TransformFn パッケージ構成と依存関係を確認し、移植先モジュールの配置図を作成する（必要なら `doc/` に図や補足を追記）。

### 2. データ型 (`datatype`) 整備
- [ ] `OHLCVDataRequest` / `OHLCVDataResponse` などを参考に、TransformFn 用の Pydantic / TypedDict 型を再設計する（ファイルパスを含めず、統計量・識別子中心にする）。
- [ ] 通貨ペア・インデックス列挙、`DataSpec` 系の設定モデルを TransformFn 用 `datatype` として導入し、`ExampleType` / `ExampleValue` / `Check` を付与する。

### 3. インデックス・前処理ロジックの移植
- [ ] `currency_index_calculator.py` の純粋計算関数（`calc_currency_index`, `create_currency_index_ohlc` 等）を TransformFn パッケージへ移植し、DataFrame 入力 → Series/DataFrame 出力に統一する。
- [ ] `resample_ohlcv`, `calculate_rsi`, `calculate_adx`, `calculate_recent_return`, `calculate_future_return`, `calculate_cyclic_feature`, `generate_spread`, `generate_swap_rate` などのテクニカル指標ロジックを移植し、各種 `Check` を実装する。
- [ ] `clean_data_and_align`, `generate_target` など、特徴量/ターゲット結合処理を TransformFn DAG で再利用できる形に整備する。

### 4. 学習・予測・評価ロジックの移植
- [ ] `convert_nullable_dtypes`, `TimeSeriesCrossValidator`, `LightGBMTrainer.train_fold`（モデル保存部分を除外）を移植し、TransformFn で再利用できる形にする。
- [ ] `TrainPredictService._execute_cross_validation` をベースに、ファイル書き出し無しで CV 結果・OOS 予測を返す `transformer` を実装する。
- [ ] `DataValidator` と評価ユーティリティ（`create_future_returns`, `create_lagged_gaps` など）を移植し、`Check` と評価 Transform を実装する。

### 5. BUY シミュレーション DAG 試作
- [ ] `train_predict_server/experiments/` のロジックを参考に、複数通貨の予測値から勝率の高い通貨を選択し BUY するシミュレーション Transform を作成する。
- [ ] シミュレーション結果（取引履歴・リターン・ドローダウン等）を返す `datatype` / `Check` を定義する。
- [ ] これらの Transform を連結したサンプル DAG（例: データ前処理 → 特徴量生成 → 学習 → 予測 → シミュレーション）を `apps/pipeline-app` に配置し、`ExampleValue` で動作例を提供する。

### 6. テストとドキュメント
- [ ] 各移植モジュールに対して単体テスト・結合テストを `uv run pytest` で実行できるよう追加する。
- [ ] `doc/` に移植範囲、利用方法、シミュレーション手順、既知の制約（Selenium 非対応・IO 省略など）を記載する。
- [ ] 必要に応じて `Makefile` / `pyproject.toml` に新しいテスト・audit コマンドを追加する。

## ステップ別完了条件
- **ステップ1: 設計と準備** — `doc/ALGO_TRADE_V3_PORTING.md` に最終決定事項の追記が行われ、移植先モジュール構成図または説明が `doc/`（例: `doc/ALGO_TRADE_V3_PORTING.md` 追記）に確認できること。
- **ステップ2: データ型整備** — 新設／更新した `datatype` 定義が `uv run mypy packages` でエラー無く型検査を通過し、型の Example/Check を利用した最小テストが `uv run pytest packages/...` で成功すること。
- **ステップ3: インデックス・前処理ロジック移植** — 移植した各関数に対する単体テストが追加され `uv run pytest`（対象モジュール）で成功、リサンプリングや指標計算の出力が既存実装と同値であることをテスト比較で担保すること。
- **ステップ4: 学習・予測・評価ロジック移植** — CV 分割・LightGBM 学習・評価モジュールのテストが `uv run pytest` で成功し、モデル保存を伴わない実装で `uv run mypy` が通ること。`train_predict_server` 既存データを用いたスモークテストで RMSE 等が算出されること。
- **ステップ5: BUY シミュレーション DAG 試作** — サンプル DAG を `uv run python -m xform_auditor apps/pipeline-app/pipeline_app --node buy_simulation` などで監査し、シミュレーション Transform の `Check` が PASS すること。Example データを用いた BUY 結果が `tests/` に記録されること。
- **ステップ6: テストとドキュメント** — `uv run pytest`（全体）と `uv run mypy packages apps`、`uv run python -m xform_auditor apps/pipeline-app/pipeline_app --format json` が成功し、生成された監査レポートの抜粋またはリンクが `doc/` 更新または PR 説明で確認できること。ドキュメントの更新差分がレビュー可能であること。

## リスク・検討事項
- LightGBM モデル保存／ロード機能を外すため、再学習前提になる点をチームに共有する必要がある。
- IO を排除することで実利用時のデータ連携方法が変わるため、別途データ供給モジュールを検討する必要がある。
- シミュレーション Transform は Example Data が小規模であることを前提とする。計算量が増える場合、分割実行やサンプリング戦略を準備する。

## 完了条件
- `make check` が成功し、`uv run pytest`（移植対象モジュールのテスト）が全て成功する。
- `uv run python -m xform_auditor apps/pipeline-app/pipeline_app` を実行し、移植した Transform 群に対する監査が `OK` となる。
- `uv run python -m xform_auditor apps/pipeline-app/pipeline_app --format json` の結果にシミュレーション Transform の評価が含まれ、BUY シナリオに対する `Check` が全て成功する（buy シミュレーション結果に対して最低限の勝率/ドローダウンチェックが PASS すること）。
- 上記 `audit` コマンドの出力例（成功ログ）を `doc/` もしくは PR 説明に添付し、客観的に確認できる状態にする。
- `doc/` に移植範囲・評価パイプライン・制約が記載され、オンボーディング時に 30 分以内でサンプル DAG を監査・実行できる手順が明示されている。

## 関連資料
- `doc/ALGO_TRADE_V3_PORTING.md`
- `algo_trade_v3/ohlcv_loader/`, `algo_trade_v3/ohlcv_preprocessor/`, `algo_trade_v3/train_predict_server/`
- `algo_trade_v3/train_predict_server/experiments/`
