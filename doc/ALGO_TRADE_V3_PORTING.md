# algo_trade_v3 移植調査メモ

TransformFn へ段階的に移植する際に、`algo_trade_v3` 配下の 3 つのサブプロジェクトから再利用できそうな構造・処理を整理した。データ読み込み → 前処理 → 特徴量・ターゲット生成 → 学習と予測 → 評価の流れに沿って候補をまとめる。

## アーキテクチャ方針（2025-10-15更新）

### パッケージ構成
```
apps/algo-trade-app/
├── algo_trade_dtype/     # アプリ固有の型・例・チェック・登録ロジック
│   ├── types.py          # TypedDict、Enum、Pydantic モデル定義
│   ├── examples.py       # ExampleValue 用の仕様オブジェクト・生成関数
│   ├── checks.py         # Check 関数群
│   └── registry.py       # RegisteredType による型登録
└── algo_trade_app/       # Transform 群・DAG 定義
    ├── transforms/       # @transform 関数群
    └── dag.py            # パイプライン定義
```

**重要**: `proj-dtypes` パッケージは廃止。各アプリケーションが独自の dtype パッケージを持ち、`xform-core` の `RegisteredType` API を使って宣言的に型を登録する。

### 移植ポリシー
- **IO 主体の処理は TransformFn へ移植しない**: `algo_trade_v3` でのファイル操作はほぼキャッシュ目的であり、TransformFn レポジトリ側には透過的なキャッシュ機構を別途実装予定。IO やキャッシュ管理そのものは `apps` 層へ持ち込まず、必要に応じて「中間データの型」だけ `algo_trade_dtype` として再定義する。
- **Selenium を用いた取得処理は今回対象外**: `ohlcv_loader` の GMO クリック証券ダウンロードは環境依存性が高いため、当面は TransformFn DAG の外に置く。
- **pyarrow / parquet 書き出しは TransformFn 側キャッシュで代替**: 既存コードの parquet 生成はキャッシュ用途。TransformFn では公式キャッシュで同等機能を提供し、必要なら parquet 相当の型定義のみ残す。
- **LightGBM は素直に対応**: 学習データが小規模であれば計算負荷は問題にならないため、`train_predict_server` の学習・推論ロジックはシンプルに移植する前提とする。
- **評価・シミュレーションは `experiments/` を参照**: `@algo_trade_v3/train_predict_server/experiments/` にある複数通貨（インデックス指数を含む）から最大期待収益の通貨を選んで BUY するシミュレーションなどを再現できるよう、評価関数やパイプライン設計に注意する。
- **Example / Check は軽量に維持**: TransformFn での `ExampleValue` や `Check` は最小限のデータ提示と値域確認に留め、`algo_trade_v3` 由来の重い再計算を持ち込まない。仕様オブジェクト（例: `HLOCVSpec(n=1000)`）で十分。複雑な検証は Transform 本体やテストで扱う。
- **`xform-auditor` CLI による自動テスト**: 型注釈のみから **入力生成 → 関数実行 → 出力検証** を自動化。**pytest を最小限に抑え**、型注釈ベースのテストで品質を担保する。


## 調査スコープ
- `algo_trade_v3/ohlcv_loader`
- `algo_trade_v3/ohlcv_preprocessor`
- `algo_trade_v3/train_predict_server`

---

## 1. データ読み込み (`ohlcv_loader`)
### algo_trade_dtype 候補
- `algo_trade_v3/ohlcv_loader/schema.py`
  - `OHLCVDataRequest`: 取得対象通貨・保存先・リトライなどを保持する要求スキーマ。TransformFn では入力 `Annotated` の `ExampleValue` に対応させやすく、キャッシュキー生成にも流用可能。
  - `OHLCVDataResponse`: 取得結果（成功/失敗通貨、処理時間、警告）を集約。返り値 TypedDict のたたき台になる。
  - `DirectoryConfig` / `DownloadConfig` / `AuthConfig`: 取得パラメータを分離した補助設定。TransformFn の型として `algo_trade_dtype/types.py` へ再定義。
  - `DataProvider`, `CurrencyPair`, `CurrencyIndex`: 列挙型。`RegisteredType` で登録し、`ExampleType` での制約に使える。
  - `validate_currency_list`, `validate_directory_path` などのバリデータ: `algo_trade_dtype/checks.py` へ移植。

### Transform 候補
- **IO 付き関数は移植対象外**: `get_ohlcv_data_paths`, `OHLCVDataService.process_request`, `_download_gmo_data`, `_unzip_files`, `_save_dataframes`, `_load_ohlc_df`, `_collect_currency_paths` は副作用の大半がキャッシュ目的のため、そのまま TransformFn へは持ち込まない。必要なら `apps` 層外部のユーティリティとして残し、TransformFn には取得済みデータを渡す想定。
- `currency_index_calculator.py` の純粋計算関数群（例: `calc_currency_index`, `create_currency_index_ohlc`）は、入力 DataFrame さえあればファイル出力なしで利用できるため `@transform` 候補。

### 移植メモ
- `OHLCVDataResponse` など Path 情報を持つ型は、TransformFn ではファイル生成を伴わない形にリファイン（例: データフレーム ID や統計量のみにする）。
- Selenium ベースのダウンロードを扱う場合は別モジュールとしてラップし、TransformFn DAG からは切り離す。
- `RegisteredType` を使って `DataProvider`, `CurrencyPair` などを宣言的に登録し、型補完を可能にする。

---

## 2. 前処理 (`ohlcv_preprocessor/src`)
### algo_trade_dtype 候補
- `algo_trade_v3/ohlcv_preprocessor/src/schema.py`
  - `FXDataSchema`: 必須/任意カラムと型検証をまとめたスキーマ。`DataValidationResult` とセットで TransformFn の入力検証へ転用。
  - `DataValidationResult`: バリデーション結果の Pydantic モデル。`Check` 側での利用も想定。
  - `Frequency`, `Currency`, `ConvertType`, `CyclicParameter`: 列挙型。`RegisteredType` で登録し、`Annotated` メタ情報に組み込み可能。
  - `DataSpec`, `TargetSetting`, `OutputConfig`, `FeatureGenerationRequest`: 特徴量生成の設定モデル。TransformFn DAG パラメータのベース。

### Transform 候補
- **IO を伴う関数は除外**: `load_currency_data_from_dir`, `get_resampled_ohlcv` などファイル読み込み＋キャッシュ前提の処理は TransformFn では利用しない。代わりに、既に手元にある DataFrame を受け取り処理する純粋関数を抜き出す。
- `resample_ohlcv` / `resample_ohlcv_df`: DataFrame を受け取ってリサンプリングする純粋処理。`@transform` に適合しやすい。
- `validate_input_data_df` / `validate_input_data`: DataFrame / ファイルバリデーション。ファイル読み込み部分を切り離し、DataFrame 版を `@transform` に取り込む。
- `get_currency_index_info`: インデックス構成情報を返す純粋計算。`algo_trade_dtype` / `@transform` 両面で活用可能。
- `clean_data_and_align`: 特徴量・ターゲットを共通インデックスに揃え統計を返す。複数入力を束ねる `@transform` として重要。

### 移植メモ
- グローバルキャッシュ `_ohlcv_cache` / `_currency_index_cache` への依存は排除し、TransformFn のキャッシュ機構へ置き換える。
- リサンプリングで parquet 書き出しは行わず、DataFrame をそのまま戻す形にする。

---

## 3. 特徴量 + ターゲット生成 (`ohlcv_preprocessor/src/service.py`)
### algo_trade_dtype 候補
- `DataSpec.feature_name`: 特徴量命名規則をカプセル化。TransformFn 設計時に列名仕様を一元管理。
- `GenerationResult`: 出力情報を保持。TransformFn ではファイルパスを外し、生成済みデータ ID やメタ情報に寄せる。

### Transform 候補
- テクニカル指標・派生値計算群（`calculate_rsi`, `calculate_adx`, `calculate_recent_return`, `calculate_future_return`, `calculate_cyclic_feature`, `generate_spread`, `generate_swap_rate`）。いずれも純粋計算なので TransformFn の小さな `@transform` として直接移植可能。
- `generate_target`: `ConvertType` に応じてターゲット/方向ラベルを生成。
- `clean_data_and_align`: 前処理と同様に重要な結合ステップ。
- `create_data`, `generate_features_with_cache`, `process_feature_generation_request`: これらはファイル IO + キャッシュを伴うラッパーのため TransformFn へは直接移植しない。内包されている純粋計算部分を分解して `@transform` 化する。

### 移植メモ
- `FeatureCacheManager` 依存部分は TransformFn キャッシュへ移行し不要化。
- 戻り値は DataFrame や Series ではなく TypedDict + `Check` を組み合わせた構造で表現する想定。

---

## 4. 学習と予測 (`train_predict_server/src`)
### algo_trade_dtype 候補
- `algo_trade_v3/train_predict_server/src/schema.py`
  - `TrainPredictRequest`: データパス・CV・LGBM 設定まとめ。TransformFn ではファイルパスを直接扱わず、データ ID や構造を受け取る設定モデルにリファイン。
  - `SimpleCVConfig`, `SimpleLGBMParams`, `SimpleOutputConfig`: デフォルトを保持した設定モデル。`ExampleValue` を付けやすい。
  - `DataPaths`: TransformFn ではファイル実体を扱わないため、データフェッチ用の識別子型に読み替える想定。
  - `PredictionResult`, `TrainPredictResponse`, `ValidationResult`: 学習結果の構造を表すモデル。戻り値 TypedDict のベース。

### Transform 候補
- `convert_nullable_dtypes`: pandas の nullable dtype を通常 dtype へ変換する純粋処理。そのまま移植。
- `TimeSeriesCrossValidator.get_cv_splits` および `_time_series_split` / `_expanding_window_split` / `_sliding_window_split`: インデックス配列を返す純粋計算。TransformFn 上で再利用可能。
- `LightGBMTrainer.train_fold`: Fold 単位の学習・評価（RMSE）。モデル保存は行わず、`Booster` とスコア/重要度を戻すように調整する。
- `TrainPredictService._execute_cross_validation`: 特徴量・ターゲットを受けて CV を実行し、OOS 予測と fold 結果を返す。ファイル書き出し `_save_oos_predictions` は削除し、DataFrame/TypedDict で返す。
- `TrainPredictService.process_train_predict_request`: IO を排除し、事前に提供されたデータ構造をもとに学習 → 評価 → シミュレーションに繋がる高レベル `@transform` へ改修。

### 移植メモ
- 学習結果を `output/` に保存する代わりに、TransformFn のキャッシュ/監査機能へ記録。
- `_calculate_score` は RMSE を返す。必要に応じて目的関数別に切り替え可能な評価モジュールへ昇格。

---

## 5. 評価・シミュレーション
### algo_trade_dtype / Transform 候補
- `ValidationResult` (`train_predict_server/src/schema.py`): 欠損数・相関などデータ品質指標。`Check` で活用。
- `DataValidator.validate_dataframe` / `.analyze_feature_target_correlation`: 特徴量・ターゲットの品質検証。`@transform` で評価ノード化。
- `TrainPredictResponse.mean_cv_score` / `.std_cv_score`: CV 結果のサマリ計算。`Check` 用の薄いラッパーとして利用。
- `reversal_modeling_validation.py` の評価ユーティリティ（`create_future_returns`, `create_lagged_gaps` など）: シミュレーション用途の純粋計算。TransformFn の評価 DAG へ組み込み、`experiments/` にある「複数通貨から最大期待収益の通貨を BUY,SELL する」シナリオを再現できるようにする。
- シミュレーション系 Transform は、複数通貨・通貨インデックスを入力とし、予測結果をランキング → 上位買い付け → リターン計算 のステップに分解して設計する。

### 移植メモ
- 評価指標（R², RMSE, MAE, Spearman, Kendall など）は `reversal_modeling_validation.py` の dataclass を参考に TypedDict 化。
- シミュレーション結果には、対象通貨、購入タイミング、想定リターン、評価ウィンドウなどを含める。

---

## 横断的な考慮事項
- **キャッシュと副作用の分離**: 既存コードが担っていたキャッシュ/ファイル保存は TransformFn の公式キャッシュへ集約し、`@transform` は純粋計算（あるいは最小限の副作用）に留める。
- **入出力ディレクトリ**: どうしてもファイルを扱う必要がある場合は `output/` 配下に限定し、TransformFn の監査機能と整合させる。
- **外部依存**: `selenium` は対象外、`pyarrow` は TransformFn キャッシュで代替、`lightgbm` は軽量データ前提でそのまま利用。
- **型付け**: Pydantic / Enum を `algo_trade_dtype` に落とし、`RegisteredType` で宣言的に登録。`Annotated` へ `ExampleType` / `Check` を付与。評価関数は `Check` として組み込む。
- **評価パイプラインの拡張性**: `experiments/` のシミュレーションケースを想定し、複数通貨・通貨インデックスを同時に扱える DAG を構築する。BUY シミュレーションの戻り値は、取引履歴・P/L・ドローダウン等を含む構造にする。

この整理をベースに、TransformFn の DAG 設計では「副作用を持たない純粋計算ノード」と「評価/シミュレーションノード」を中心に据え、IO 依存はリポジトリ外の仕組みへ切り出す方針を徹底する。

---

## 移植フェーズと完了条件

### Phase 1: 基礎型・例・チェック整備

#### 実装内容
- `apps/algo-trade-app/algo_trade_dtype/` パッケージの作成
  - `types.py`: OHLCV データ型、Enum（Frequency, Currency など）、設定モデル定義
  - `examples.py`: HLOCVSpec など仕様オブジェクト定義と生成関数
  - `checks.py`: DataFrame 構造・値域・欠損チェック関数群
  - `registry.py`: RegisteredType による型登録とメタデータバインド

#### 完了条件
```bash
# 静的検査が成功（make check に含まれる）
make check
# => duplication, format, lint, typecheck, complexity すべて成功
```

**NOTE**: Phase 1 では `@transform` 関数が未実装のため `audit` コマンドは実行不可。型定義・Example・Check 関数の品質は `make check` で担保。必要に応じて `checks.py` 内の個別チェック関数の動作確認用 pytest を最小限追加可能だが、Phase 2 以降で `audit` により検証されるため必須ではない。

---

### Phase 2: 前処理・特徴量 Transform 移植

#### 実装内容
- `apps/algo-trade-app/algo_trade_app/transforms/preprocessing.py`
  - `resample_ohlcv`: リサンプリング Transform
  - `calc_currency_index`: 通貨インデックス計算
  - `validate_ohlcv`: DataFrame 検証
- `apps/algo-trade-app/algo_trade_app/transforms/features.py`
  - `calculate_rsi`, `calculate_adx`, `calculate_recent_return` など
  - `generate_target`: ターゲット生成
  - `clean_data_and_align`: 特徴量・ターゲットの結合

#### 完了条件（audit 実行結果）
```bash
# audit CLI による前処理・特徴量 Transform の検証
uv run python -m xform_auditor algo_trade_app.transforms.preprocessing
uv run python -m xform_auditor algo_trade_app.transforms.features
```

**期待される出力例**:
```
Auditing algo_trade_app.transforms.preprocessing...
  [OK] resample_ohlcv (Example: HLOCVSpec(n=1000, freq='1min') → Check: check_ohlcv passed)
  [OK] calc_currency_index (Example: multi-currency OHLCV → Check: check_currency_index passed)
  [OK] validate_ohlcv (Example: HLOCVSpec(n=100) → Check: check_validation_result passed)

Auditing algo_trade_app.transforms.features...
  [OK] calculate_rsi (Example: HLOCVSpec(n=100) → Check: check_rsi_series passed)
  [OK] calculate_adx (Example: HLOCVSpec(n=100) → Check: check_adx_series passed)
  [OK] generate_target (Example: HLOCVSpec(n=1000) → Check: check_target_labels passed)
  [OK] clean_data_and_align (Example: features + target → Check: check_aligned_data passed)

Summary: 7 transforms, 7 OK, 0 VIOLATION, 0 ERROR, 0 MISSING
```

---

### Phase 3: 学習・予測 Transform 移植

#### 実装内容
- `apps/algo-trade-app/algo_trade_app/transforms/training.py`
  - `convert_nullable_dtypes`: nullable dtype 変換
  - `get_cv_splits`: CV 分割インデックス生成
  - `train_fold`: Fold 単位の学習・評価
  - `execute_cross_validation`: CV 実行と OOS 予測生成

#### 完了条件（audit 実行結果）
```bash
uv run python -m xform_auditor algo_trade_app.transforms.training
```

**期待される出力例**:
```
Auditing algo_trade_app.transforms.training...
  [OK] convert_nullable_dtypes (Example: nullable DataFrame → Check: check_dtype_conversion passed)
  [OK] get_cv_splits (Example: SimpleCVConfig(n_splits=5) → Check: check_cv_splits passed)
  [OK] train_fold (Example: features + target + fold_idx → Check: check_fold_result passed)
  [OK] execute_cross_validation (Example: full dataset + CVConfig → Check: check_cv_result passed)

Summary: 4 transforms, 4 OK, 0 VIOLATION, 0 ERROR, 0 MISSING
```

#### 補足

Phase 3 の train_fold Transform は、戻り値として lightgbm.Boosterオブジェクトを返す設計になっています。BoosterオブジェクトはそのままではJSON等にシリアライズできないため、TransformFn のデフォルトキャッシュ機構で扱う際に、特別なシリアライズ・デシリアライズ処理（materialization）が必要になる可能性があります。Booster オブジェクトのキャッシュには xform-core の materialization機構の活用を検討する

---

### Phase 4: 評価・シミュレーション Transform 移植

#### 実装内容
- `apps/algo-trade-app/algo_trade_app/transforms/simulation.py`
  - `rank_predictions`: 予測結果のランキング生成
  - `select_top_currency`: 最大期待収益通貨の選択
  - `simulate_buy_scenario`: BUY シミュレーション実行
  - `calculate_performance_metrics`: P/L・ドローダウン計算

#### 完了条件（audit 実行結果）
```bash
uv run python -m xform_auditor algo_trade_app.transforms.simulation
```

**期待される出力例**:
```
Auditing algo_trade_app.transforms.simulation...
  [OK] rank_predictions (Example: multi-currency predictions → Check: check_ranking passed)
  [OK] select_top_currency (Example: ranked predictions → Check: check_selection passed)
  [OK] simulate_buy_scenario (Example: top currency + OHLCV → Check: check_simulation passed)
  [OK] calculate_performance_metrics (Example: simulation result → Check: check_metrics passed)

Summary: 4 transforms, 4 OK, 0 VIOLATION, 0 ERROR, 0 MISSING
```

---

### Phase 5: 統合 DAG とエンドツーエンドテスト

#### 実装内容
- `apps/algo-trade-app/algo_trade_app/dag.py`: 全 Transform を統合した DAG 定義
- エンドツーエンドシナリオの pytest 追加（定義されたサンプルのDAG実行ができて、メトリクスのチェック基準に従うデータが無事出力される正常系の動作確認があれば十分）

#### 完了条件（全体 audit + pytest）
```bash
# 全 Transform の audit 実行
uv run python -m xform_auditor algo_trade_app --format json > output/audit_result.json
```

**期待される JSON 出力構造**:
```json
{
  "summary": {
    "total": 18,
    "ok": 18,
    "violation": 0,
    "error": 0,
    "missing": 0
  },
  "transforms": [
    {
      "name": "algo_trade_app.transforms.preprocessing.resample_ohlcv",
      "status": "OK",
      "example": "HLOCVSpec(n=1000, freq='1min')",
      "check": "algo_trade_dtype.checks.check_ohlcv",
      "execution_time_ms": 45
    },
    ...
    {
      "name": "algo_trade_app.transforms.simulation.simulate_buy_scenario",
      "status": "OK",
      "example": "top_currency + OHLCV",
      "check": "algo_trade_dtype.checks.check_simulation",
      "execution_time_ms": 120
    }
  ]
}
```

```bash
# pytest による統合テスト
uv run pytest apps/algo-trade-app/tests/test_integration.py
# => 全テストパス、DAG 実行成功
```

```bash
# 品質チェック全体
make check
# => duplication, format, lint, typecheck, complexity すべて成功
```

---

## 最終成果物

### 必須ドキュメント
- `apps/algo-trade-app/README.md`: セットアップ・実行手順
- `output/audit_result.json`: 全 Transform の audit 結果（Phase 5 で生成）
- `doc/ALGO_TRADE_APP_ARCHITECTURE.md`: 移植範囲・制約・評価パイプライン詳細

### 検証項目チェックリスト
- [ ] `make check` 成功
- [ ] Phase 1-4 の各 audit コマンド実行で全て OK
- [ ] Phase 5 の全体 audit で `summary.ok == summary.total`
- [ ] `output/audit_result.json` に全 Transform の成功ログが含まれる
- [ ] シミュレーション Transform の `Check` が全て成功
- [ ] pytest 統合テストが成功
- [ ] README 手順に従い 監査・実行が完了可能
