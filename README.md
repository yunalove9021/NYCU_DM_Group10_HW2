# RF 與 GCN_RF

本專案提供兩種二元節點分類流程，支援以 YAML 設定多組特徵集與 random seed，並透過模型索引載入已訓練模型進行預測。

| 模型 | 輸入與分類方式 |
| --- | --- |
| RF | 使用指定的 node 特徵訓練 Random Forest。 |
| GCN_RF | 使用 node 特徵與 edge 訓練監督式 GCN，將 100 維 node embedding 與原特徵合併後，再訓練 Random Forest。 |

標籤欄位預設為 `y`：`0` 為正常、`1` 為異常。無標籤節點不參與分類 loss 或 RF 擬合，但可產生預測；GCN 訓練時也會保留這些節點的圖連接。

## 執行環境

於專案根目錄執行下列指令。RF 與 GCN_RF 的依賴已合併至同一份 `requirements.txt`：

```powershell
python -m pip install -r requirements.txt
```

載入既有模型時，請使用與訓練相容的套件版本，尤其是 scikit-learn 與 PyTorch。

## 訓練設定

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
```

`basic.json` 包含目前訓練資料從 index 4 起的 165 個特徵欄名。新增特徵組合時，可在 `configs/feature_sets/` 建立另一份 JSON，再加入上述清單；不同特徵集的檔名須唯一。

目前兩種模型的 Random Forest 均設定為 300 棵樹、`class_weight: balanced`。GCN 使用兩層 `input → 128 → 100`，搭配二元分類頭與 weighted cross-entropy；類別權重只依 train 的已知標籤計算，train 須同時包含兩類。

GCN 的 scaler 僅以 train 擬合，encoder 只用 train 標籤更新權重，再以 validation loss 選擇 checkpoint。依目前 YAML，最多訓練 10 epochs、patience 為 20；無驗證標籤時採最後一個 epoch。固定 encoder 後產生各資料集的 `emb_dim_1` 至 `emb_dim_100`，test 不參與訓練或選模。

GCN 預設保留 `txId1 → txId2` 方向；`undirected: true` 可加入反向邊。各 split 以整張圖批次推論，表示預測時該 split 的 node 特徵與 edge 皆已可見。

## 訓練指令

每次指令只執行所選模型。未指定 `--model` 時預設為 RF。

```powershell
python main.py --config configs/base.yaml --model RF
python main.py --config configs/base.yaml --model GCN_RF
```

RF 不需要 edge 或 encoder；GCN_RF 的 node CSV 必須有唯一且非缺失的 `txId`，edge CSV 必須有 `txId1`、`txId2`，且端點都存在於對應 node CSV。沒有 edge 的孤立節點可以正常處理。

## 模型與訓練輸出

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

## 使用 test 資料預測

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

## Python 調用介面

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

## 訓練結果 Excel 報表

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
