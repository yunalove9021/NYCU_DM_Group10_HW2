# Data Mining HW2 (Group 10)

## Dataset: Elliptic Bitcoin Dataset

本專案使用 [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) 進行 Bitcoin 非法交易辨識。資料由 Elliptic 與 MIT-IBM Watson AI Lab 發表，詳細定義見 Weber et al. 的 [原始論文](https://arxiv.org/abs/1908.02591)。

由於資料檔案大於標準 GitHub 上傳限制，我們透過 Google Drive 資料夾提供處理後的完整資料集，請於 [本連結](https://drive.google.com/drive/folders/1CP_xMwjkMYdurRGi3JEXve5pLavoa1jq?usp=drive_link) 下載整份資料後，解壓縮至本專案目錄下使用。

### 資料結構

- **Nodes：203,769**
  - 每個 node 代表一筆 Bitcoin transaction。
- **Edges：234,355**
  - Directed edge `txId1 → txId2` 表示 BTC 由前一筆交易流向下一筆交易。
- **Node features：166**
  - `time_step`
  - 93 個匿名 local transaction features
  - 72 個由一跳前後鄰居彙整而成的 aggregated features
- **Time steps：49**
  - 相鄰 time steps 約間隔兩週。
  - 每個 time step 是由發生時間相近、相差少於三小時的交易組成的獨立圖。
  - 不同 time steps 之間沒有 edges。

### 標籤分布

- **Illicit：4,545**
  - 原始標籤為 `class=1`，處理後映射為 `y=1`。
  - 包含 scam、malware、ransomware、Ponzi scheme 等非法實體所發起的交易。
- **Licit：42,019**
  - 原始標籤為 `class=2`，處理後映射為 `y=0`。
  - 包含 exchange、wallet provider、miner 等合法實體所發起的交易。
- **Unknown：157,205**
  - 保留為無標籤節點。
  - 不參與分類 loss、Random Forest 擬合或評估指標計算。
  - 在圖模型中仍保留其 node features 與 edge connections。

### 資料處理

- 使用 `txId` 合併原始 node features 與 class labels。
- 保留 `txId`、`time_step`、原始標籤與 165 個匿名數值特徵。
- Node 與 edge 資料依相同的 time-step 範圍切分。
- 採用 chronological split，避免未來時間的交易資訊進入訓練資料。

### Dark market shutdown

- 原始論文指出，`time_step 43` 發生了一次 **dark market shutdown**。
- 論文沒有公開該 dark market 的具體名稱。
- 該事件造成交易與非法活動模式改變，原始論文中的所有模型在事件後均出現明顯效能下降。
- 本專案將測試資料分為事件前後兩段，用來比較模型面對突發事件與時間分布轉移時的穩健性。

### 資料切分

| Split | Time step | Nodes | Edges |
| --- | ---: | ---: | ---: |
| Train | 1–30 | 123,287 | 142,784 |
| Validation | 31–35 | 18,485 | 20,410 |
| Test（pre-shutdown） | 36–42 | 32,313 | 37,502 |
| Test（post-shutdown） | 43–49 | 29,684 | 33,659 |

原始論文使用 time steps 1–34 訓練、35–49 測試；本專案另外保留 validation set，並將測試資料依 dark market shutdown 分為事件前後兩段。

## RF 與 GCN_RF

本專案提供兩種二元節點分類流程，支援以 YAML 設定多組特徵集與 random seed，並透過模型索引載入已訓練模型進行預測。

| 模型 | 輸入與分類方式 |
| --- | --- |
| RF | 使用指定的 node 特徵訓練 Random Forest。 |
| GCN_RF | 使用 node 特徵與 edge 訓練監督式 GCN，將 100 維 node embedding 與原特徵合併後，再訓練 Random Forest。 |

標籤欄位預設為 `y`：`0` 為正常、`1` 為異常。無標籤節點不參與分類 loss 或 RF 擬合，但可產生預測；GCN 訓練時也會保留這些節點的圖連接。

### 執行環境

於專案根目錄執行下列指令。RF 與 GCN_RF 的依賴已合併至同一份 `requirements.txt`：

```powershell
python -m pip install -r requirements.txt
```

載入既有模型時，請使用與訓練相容的套件版本，尤其是 scikit-learn 與 PyTorch。

### 訓練設定

訓練使用 [configs/base.yaml](configs/base.yaml)，各模型參數獨立設定。

| 設定 | 說明 |
| --- | --- |
| `input`、`output` | 資料與訓練輸出目錄，相對於 YAML 所在目錄。 |
| `random_seeds` | 單一整數或多組 seed；每組特徵集皆會執行所有 seed。 |
| `data.train_file`、`data.val_file` | 訓練與驗證 CSV，檔名相對於 `input`。 |
| `data.test_files` | 多組測試 CSV，以資料集名稱對應檔名。 |
| `data.edge_files` | GCN_RF 的 edge CSV，名稱須對應 `train`、`val` 與各測試資料集。 |
| `data.feature_columns` | 一個或多個特徵 JSON 路徑，相對於 YAML 所在目錄。 |
| `models.RF.params` | RF 的 Random Forest 參數。 |
| `models.GCN_RF.params` | GCN_RF 後段的 Random Forest 參數。 |
| `models.GCN_RF.encoder` | GCN 架構、訓練參數、loss、裝置與 edge 方向設定。 |

特徵 JSON 為依序排列的欄名清單，檔名作為特徵集名稱。例如：

```yaml
data:
  feature_columns:
    - feature_sets/basic.json
    - feature_sets/local_features.json
    - feature_sets/feature_selection_cor_095.json
    - feature_sets/feature_selection_cor_09.json
```

`basic.json` 包含目前訓練資料從 index 4 起的 165 個特徵欄名。新增特徵組合時，可在 `configs/feature_sets/` 建立另一份 JSON，再加入上述清單；不同特徵集的檔名須唯一。

目前兩種模型的 Random Forest 均設定為 300 棵樹、`class_weight: balanced`。GCN 使用兩層 `input → 128 → 100`，搭配二元分類頭與 weighted cross-entropy；類別權重只依 train 的已知標籤計算，train 須同時包含兩類。

GCN 的 scaler 僅以 train 擬合，encoder 只用 train 標籤更新權重，再以 validation loss 選擇 checkpoint。依目前 YAML，最多訓練 10 epochs、patience 為 20；無驗證標籤時採最後一個 epoch。固定 encoder 後產生各資料集的 `emb_dim_1` 至 `emb_dim_100`，test 不參與訓練或選模。

GCN 預設保留 `txId1 → txId2` 方向；`undirected: true` 可加入反向邊。各 split 以整張圖批次推論，表示預測時該 split 的 node 特徵與 edge 皆已可見。

### 訓練指令

每次指令只執行所選模型。未指定 `--model` 時預設為 RF。

```powershell
python main.py --config configs/base.yaml --model RF
python main.py --config configs/base.yaml --model GCN_RF
python main.py --config configs/base.yaml --model random_score
python main.py --config configs/base.yaml --model single_feature
python main.py --config configs/base.yaml --model isolation_forest
python main.py --config configs/base.yaml --model AE
python main.py --config configs/base.yaml --model GAE
```

RF 不需要 edge 或 encoder；GCN_RF 的 node CSV 必須有唯一且非缺失的 `txId`，edge CSV 必須有 `txId1`、`txId2`，且端點都存在於對應 node CSV。沒有 edge 的孤立節點可以正常處理。


### 模型與訓練輸出

結果依模型與特徵集分開保存：

```text
output/
├── RF_output/<set_name>/
│   ├── model/
│   ├── eval/
│   └── predict/
└── GCN_RF_output/<set_name>/
    ├── model/
    ├── eval/
    └── predict/
```

| 資料夾 | 內容 |
| --- | --- |
| `model/` | Random Forest、實際輸入特徵表；GCN_RF 另有 encoder 與 scaler。 |
| `eval/` | Metrics、ROC 圖與曲線座標、特徵重要度、標籤摘要；GCN_RF 另有訓練紀錄。 |
| `predict/` | 驗證與測試集預測；GCN_RF 另有各資料集的 embedding CSV。 |

檔名前綴為 `<set_name>_seed_<seed>_<model_name>`。評估包含 precision、recall、F1、MCC、混淆矩陣、AUROC 與 PR-AUC（Average Precision），僅使用已知標籤；AUROC、PR-AUC 與 ROC 需要同時有兩類標籤。同名結果再次執行時會覆寫。

模型集中登錄於 [configs/model.json](configs/model.json)，每個實例各有一個索引，例如 `seed_42_RF_basic` 或 `seed_42_GCN_RF_local_features`。每筆記錄包含 `seed`、`model_name`、`set_name` 與 `path`，連到模型、特徵表，以及 GCN_RF 所需的 encoder、scaler。路徑相對於 `model.json` 所在目錄。

訓練會新增或更新對應索引，保留其他模型；若指定其他位置的 YAML，則更新該 YAML 同目錄的 `model.json`。移動模型檔案後須同步調整索引路徑。

### 使用 test 資料預測

`predict.py` 依模型索引載入已訓練模型，不會重新訓練。以下範例使用 `data/` 中事件前、事件後的測試資料。

**RF：使用 basic 特徵集、seed 42。**

```powershell
python predict.py --model RF --index seed_42_RF_basic --input data/nodes_test_pre_event.csv
python predict.py --model RF --index seed_42_RF_basic --input data/nodes_test_post_event.csv
```

**GCN_RF：使用 local_features 特徵集、seed 42，並傳入同時段的 edge。**

```powershell
python predict.py --model GCN_RF --index seed_42_GCN_RF_local_features --input data/nodes_test_pre_event.csv --edge data/edge_test_pre_event.csv
python predict.py --model GCN_RF --index seed_42_GCN_RF_local_features --input data/nodes_test_post_event.csv --edge data/edge_test_post_event.csv
```

| 參數 | 說明 |
| --- | --- |
| `--model` | 必填，`RF` 或 `GCN_RF`。 |
| `--index` | 必填，`model.json` 中的完整索引，須與模型類型一致。 |
| `--input` | 必填，待預測的 node CSV。 |
| `--edge` | GCN_RF 必填，對應 node 資料的 edge CSV；RF 不使用。 |
| `--config` | 模型索引 JSON，預設為專案的 `configs/model.json`。 |
| `--device` | GCN 推論裝置，預設 `cpu`，可指定 `cuda`。 |

CSV 路徑相對於執行時的工作目錄，也可傳入絕對路徑。更換 seed 或特徵集時，請使用 `model.json` 中存在的完整索引。

輸入不需包含 `y`，但須具備該模型全部必要特徵，且特徵須為數值、無缺失或無限值。GCN_RF 會自動產生 embedding，不需在 input CSV 預先加入 `emb_dim_1..100`。模型類型不符、缺少特徵、未傳必要 edge 或 edge 端點不存在時，程式會報錯。

預測結果固定存於專案根目錄的 `predict_result/`，例如：

```text
predict_result/seed_42_RF_basic_nodes_test_pre_event_predictions.csv
predict_result/seed_42_GCN_RF_local_features_nodes_test_post_event_predictions.csv
```

CSV 保留輸入順序，包含 `input_row`（從 0 起）、可用的 `txId` 與 `time_step`、`y_pred`、`probability_normal`、`probability_anomaly`。同索引、同輸入檔名的預測會覆寫原結果。

### Python 調用介面

需要在其他 Python 程式中調用時，可由索引載入模型與特徵順序：

```python
import pandas as pd
from src.utils.artifacts import load_artifacts, prepare_model_input

bundle = load_artifacts("configs/model.json", "seed_42_GCN_RF_basic")
nodes = pd.read_csv("data/nodes_test_pre_event.csv")
edges = pd.read_csv("data/edge_test_pre_event.csv")
x = prepare_model_input(bundle, nodes, edges)
predictions = bundle["model"].predict(x)
```

RF 使用相同介面，改選 RF 索引並呼叫 `prepare_model_input(bundle, nodes)` 即可。

### 訓練結果 Excel 報表

```powershell
python analysis_check.py --config configs/base.yaml
```

程式依 YAML 中的模型、特徵集、seed 與資料集查詢 output，不需要重新訓練或載入模型。
報表存至 `Analysis/training_results.xlsx`，以模型名稱排序，包含：

| 工作表 | 內容 |
| --- | --- |
| `Metrics_by_seed` | 各模型、特徵集、資料集與 seed 的 metrics，包含混淆矩陣各值。 |
| `Seed_summary` | 同組結果跨 seed 的平均、樣本標準差，以及整組預期與已讀取 seed 數。 |
| `Hyperparameters` | 目前 YAML 的 RF／GCN 超參數、設定值與用途說明。 |
| `Run_status` | 各預期結果是否存在、可讀取，以及缺失原因。 |
| `Notes` | 設定來源、統計規則與欄位說明。 |

平均值不混合不同模型、特徵集或資料集；各指標以非缺失值計算，不當成 0。
seed 數僅在每組結果列出一次，個別指標是否缺值可回查 `Metrics_by_seed`。
超參數描述來自目前 YAML，若訓練後修改設定，可能與歷史結果不同。
可用 `--output Analysis/其他檔名.xlsx` 指定報表檔名。

### RF／GCN_RF 特徵視覺化

`visualize_features.py` 讀取指定訓練 output，僅分析 RF 與 GCN_RF，其他模型資料夾不會納入。
原始資料只讀取 `base.yaml` 指定的 train node 與 train edge，不讀取 validation／test 的特徵或標籤。

```powershell
python visualize_features.py --config configs/base.yaml --output-dir output
```

預設按 YAML 的 5 個 seed，分析 output 下的全部特徵集。也可指定特徵集、seed 與儲存位置：

```powershell
python visualize_features.py --output-dir output --feature-sets basic local_features --seeds 42 123 456 789 1105 --top 20 --save-dir Analysis/feature_visualization
```

每個模型／特徵集都必須有完整 5 組 `*_feature_importance.csv`，且各 seed 的特徵空間一致。
缺少 seed 或欄位時會直接報錯，不會將缺失當成 0。若要刻意分析其他數量，需明確設定
`--expected-seeds` 並傳入相同數量的 seed。可用 `--models RF` 僅分析 RF；`GCN-RF` 也可作為 `GCN_RF` 的輸入名稱。

結果預設存於 `Analysis/feature_visualization/`，每張圖同時提供 PNG 與 PDF，並附 CSV 數值表：

| 圖表 | 內容與計算方式 |
| --- | --- |
| `<特徵集>/01_mi_ks_rankings` | MI 與 KS distance 並排 Top 20 水平長條圖；僅使用有標籤的 train nodes，保留原始類別比例。 |
| `<特徵集>/02_spearman_clustered` | Top 20 特徵的 Spearman heatmap，使用全部 train nodes（含無標籤節點）的完整資料列，按 `1 - abs(rho)` 做 average-linkage 階層分群。 |
| `<特徵集>/03_importance_stability` | RF、GCN_RF 分開 panel，呈現 5 seed 平均 impurity importance、樣本 SD，以及各 seed 進入 Top 10 的次數。 |
| `04_graph_structural_ecdf` | Train t1–30 的 in-degree、out-degree、鄰居平均 degree、local clustering 類別條件 ECDF，標示 MI／KS。 |

Spearman 預設從 MI、KS、RF importance 排名交錯選取特徵，優先保留
`feature_5`、`feature_18`、`feature_60`、`feature_90`。如果這四個欄位不在該組候選特徵中，
仍可作為 train 參考欄位加入，圖上以 `*` 標記；焦點特徵以紅色標示。
可用 `--heatmap-source mi`、`ks` 或 `rf` 改變選取來源；`--top` 支援 15–20。
高相關門檻預設為 `abs(rho) >= 0.9`，可用 `--correlation-threshold` 調整。

MI 以 nats 表示，原始候選欄位預設為連續變數，使用 kNN estimator（`n_neighbors=3`、random state 42）；
已知離散欄位可用 `--discrete-features` 指定。KS 輸出兩類 ECDF 的最大距離 D，不以 p-value 進行顯著性判斷。
Spearman 不納入 embedding，僅檢查原始 train 欄位的冗餘關係。常數欄位與缺失值處理會記錄於數值表或分析資訊。

GCN_RF panel 的 importance 來自後段 RF，與 RF panel 都是 mean decrease in impurity，並非 GCN 權重。
原始特徵與 embedding 使用不同顏色。不同 seed 的 GCN embedding 座標未對齊，
單一 `emb_dim_*` 的跨 seed 波動僅作描述，不代表每個座標具有相同語意。

圖結構特徵由 train t1–30 的全部節點與 edge 建立，再使用已知 train 標籤比較 licit／illicit。
移除重複 edge 與 self-loop；in/out degree 保留方向，鄰居平均 degree 與 clustering 使用無向投影。
不將未知語意的 `feature_*` 當成交易金額，也不使用任何測試標籤計算圖特徵。

`focus_spearman.csv` 保存四個焦點特徵的相關矩陣，`analysis_manifest.json` 記錄資料來源、
seed、方法、高相關配對與樣本數；其餘 CSV 包含完整 MI／KS 排名、各 seed importance、穩定度彙整與圖結構特徵。

### Validation threshold 分析

`threshold_check.py` 只讀取 output 中 RF／GCN_RF 已保存的預測 CSV，不重新訓練或讀取原始資料。
預設以 `0.01` 步距掃描 `0.50` 至 `1.00`（含端點），依 validation 的 anomaly F1 選擇 threshold，
選定後再套用至 validation 與各 test；test 標籤不參與 threshold 選擇。

```powershell
python threshold_check.py --config configs/base.yaml --output-dir output
python threshold_check.py --output-dir output --feature-sets basic local_features --seeds 42 123 456 789 1105
```

每組模型／特徵集先將 5 個 seed 的預測依 `txId` 對齊，再逐筆計算
`mean(probability_anomaly)`，形成一份 ensemble 預測。節點集合、標籤與 `time_step`
必須一致，列順序可不同。以平均後的 validation 機率計算 F1 並選出單一 threshold，
再套用到同樣平均後的 val／test 機率；不單獨計算各 seed 的指標，也不平均各 seed 的 F1。
若 F1 完全相同，採最低 threshold。判定規則為 `mean(probability_anomaly) >= threshold`。

結果分開存於 `Analysis/threshold/<模型>/<特徵集>/`：

| 檔案 | 內容 |
| --- | --- |
| `validation_threshold_curves.png`／`.pdf` | 平均機率 ensemble 的 validation anomaly F1 折線與 PR-AUC 參考線。 |
| `validation_threshold_scan.csv` | Ensemble 在各 threshold 的 validation metrics，每個 threshold 只有一筆。 |
| `best_threshold_results.csv` | Ensemble 在最佳 threshold 的 val／test 結果，包含 precision、recall、F1、MCC、AUROC、PR-AUC、混淆矩陣與筆數。 |
| `threshold_results.xlsx` | `Val_scan`、`Best_results`、`Ensemble_info` 三個工作表。 |
| `threshold_selection.json` | 最佳 threshold、probability averaging 規則、seed 與輸入檔案路徑。 |

PR-AUC 沿用 Average Precision 定義，由平均後的機率計算，**不隨 threshold 改變**，
因此在 threshold 圖上呈水平線；最佳 threshold 依 F1 選取，不會將二值化預測當成 AP 的機率輸入。
僅使用已知標籤計算指標，沒有標籤的 test 仍可另存預測，metrics 則留白。

可透過 `--start`、`--end`、`--step` 調整掃描範圍，或 `--models RF` 只分析 RF。
預設要求 5 個 seed 齊全；刻意使用其他數量時需設定 `--expected-seeds`，至少為 2。
加上 `--save-predictions` 會額外輸出最佳 threshold 下的平均機率與逐筆 val／test 預測 CSV；
預設只輸出圖表與指標，保留原 output 預測檔不變。舊版的個別 seed threshold 與平均 metrics 檔會在成功更新後移除。
