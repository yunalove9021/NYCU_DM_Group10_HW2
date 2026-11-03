import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- 1. 新增 PyTorch Geometric 的 GCN 卷積層 ---
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    matthews_corrcoef
)
from sklearn import set_config
set_config(display="text")
random_seed = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- 2. 新增建立 edge_index 的函數 ---
def create_edge_index(nodes_df, edges_df):
    if nodes_df["txId"].duplicated().any():
        raise ValueError("Duplicate txId found in nodes_df.")
    node_ids = nodes_df['txId'].values
    id2idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    valid_edges = edges_df[
        edges_df['txId1'].isin(id2idx) &
        edges_df['txId2'].isin(id2idx)
    ]
    source = valid_edges['txId1'].map(id2idx).to_numpy(dtype=np.int64)
    target = valid_edges['txId2'].map(id2idx).to_numpy(dtype=np.int64)
    if len(source) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.from_numpy(np.stack([source, target], axis=0))
    # edge_index = to_undirected(edge_index) # Undirected TODO
    return edge_index

# --- 修改評估函數，加入 edges 參數 ---
def evaluate_and_save(model, data, edges, features_name, name, output):
    """輸出所有資料列的預測，僅用有標籤的資料計算評估指標。"""
    
    # 呼叫預測時傳入 nodes_df, edges_df, 與 features
    y_score = model.predict_score(data, edges, features_name)
    if model.threshold is None:
        raise ValueError("Threshold has not been selected.")
    if model.reverse_threshold is None:
        raise ValueError("Reversed threshold has not been selected.")
    # Original: MSE
    original_score = y_score

    # Reversed: -MSE
    reversed_score = -y_score

    original_pred = (
        original_score >= model.threshold
    ).astype(int)

    reversed_pred = (
        reversed_score >= model.reverse_threshold
    ).astype(int)
    
    known = data["y"].notna()

    actual = data["y"]

    pd.DataFrame({
        "time_step": data["time_step"],
        "y_true": actual,
        "y_pred": original_pred,
        "anomaly_score": original_score,
    }).to_csv(
        output / f"{name}_original_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 輸出 Reversed Predictions
    pd.DataFrame({
        "time_step": data["time_step"],
        "y_true": actual,
        "y_pred": reversed_pred,
        "anomaly_score": reversed_score,
    }).to_csv(
        output / f"{name}_reversed_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "n_rows": int(len(data)),
        "n_labeled": int(known.sum()),
        "original": {},
        "reversed": {},
    }
    curve_data = None

    if known.any():
        y_true = actual.loc[known].astype(int).to_numpy()

        original_score_labeled = original_score[known.to_numpy()]
        reversed_score_labeled = reversed_score[known.to_numpy()]

        original_pred_labeled = original_pred[known.to_numpy()]
        reversed_pred_labeled = reversed_pred[known.to_numpy()]

        def calculate_metrics(y_true, y_pred, y_score):
            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            result = {
                "precision_anomaly": float(
                    precision_score(
                        y_true, y_pred,
                        pos_label=1,
                        zero_division=0
                    )
                ),
                "recall_anomaly": float(
                    recall_score(
                        y_true, y_pred,
                        pos_label=1,
                        zero_division=0
                    )
                ),
                "f1_anomaly": float(
                    f1_score(
                        y_true, y_pred,
                        zero_division=0
                    )
                ),
                "mcc": (
                    float(matthews_corrcoef(y_true, y_pred))
                    if len(np.unique(y_true)) == 2
                    else 0.0
                ),
                "confusion_matrix": {
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                },
            }

            if len(np.unique(y_true)) == 2:
                result["auroc"] = float(
                    roc_auc_score(y_true, y_score)
                )

                result["pr_auc"] = float(
                    average_precision_score(y_true, y_score)
                )
            else:
                result["auroc"] = None
                result["pr_auc"] = None

            return result

        metrics["original"] = calculate_metrics(
            y_true,
            original_pred_labeled,
            original_score_labeled
        )

        metrics["reversed"] = calculate_metrics(
            y_true,
            reversed_pred_labeled,
            reversed_score_labeled
        )


        if len(np.unique(y_true)) == 2:
            fpr_original, tpr_original, thresholds_original = roc_curve(
                y_true,
                original_score_labeled,
                pos_label=1
            )

            fpr_reversed, tpr_reversed, thresholds_reversed = roc_curve(
                y_true,
                reversed_score_labeled,
                pos_label=1
            )

            pd.DataFrame({
                "fpr": fpr_original,
                "tpr": tpr_original,
                "threshold": thresholds_original,
            }).to_csv(
                output / f"{name}_original_roc_points.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame({
                "fpr": fpr_reversed,
                "tpr": tpr_reversed,
                "threshold": thresholds_reversed,
            }).to_csv(
                output / f"{name}_reversed_roc_points.csv",
                index=False,
                encoding="utf-8-sig",
            )

            curve_data = {
                "original": (fpr_original, tpr_original),
                "reversed": (fpr_reversed, tpr_reversed),
            }

            print(f"\n{name.upper()}")

            print(
                f"Original MSE: "
                f"AUROC={metrics['original']['auroc']:.4f}, "
                f"AP={metrics['original']['pr_auc']:.4f}, "
                f"F1={metrics['original']['f1_anomaly']:.4f}"
            )

            print(
                f"Reversed -MSE: "
                f"AUROC={metrics['reversed']['auroc']:.4f}, "
                f"AP={metrics['reversed']['pr_auc']:.4f}, "
                f"F1={metrics['reversed']['f1_anomaly']:.4f}"
            )

    return metrics, curve_data

def summarize_labels(data_sets, output_dir):
    output_dir = Path(output_dir)
    all_summary = []
    for data, name in data_sets:
        y = data["y"] if "y" in data.columns else pd.Series(float("nan"), index=data.index)
        summary = {
            "dataset": name,
            "total": int(len(data)),
            "labeled": int(y.notna().sum()),
            "normal_0": int((y == 0).sum()),
            "anomaly_1": int((y == 1).sum()),
            "unlabeled_nan": int(y.isna().sum()),
        }
        all_summary.append(summary)
    summary = pd.DataFrame(all_summary)
    summary.to_excel(output_dir / f"label_summary.xlsx", index=False)
    return summary

# --- 3. 將 PyTorch模型替換為 GAE 模型 ---
class GAE_Model(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(GAE_Model, self).__init__()
        # Encoder: 兩層 GCNConv
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, latent_dim)
        # Decoder: 一般的線性層即可
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, edge_index):
        z = torch.relu(self.conv1(x, edge_index))
        z = torch.relu(self.conv2(z, edge_index))
        return self.decoder(z)

# --- 模型包裹類別更新為 GraphAutoencoderModel ---
class GraphAutoencoderModel:
    def __init__(self, input_dim, latent_dim=64, epochs=50, lr=1e-3, random_seed=42):
        set_seed(random_seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GAE_Model(input_dim, latent_dim).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.threshold = None
        self.reverse_threshold = None
        self.original_best_f1 = None
        self.reverse_best_f1 = None

    def fit(self, nodes_df, edges_df, features):
        """
        防漏規範：只使用純正常節點 (y==0) 建立 Subgraph 與進行訓練
        """
        y_train = pd.to_numeric(nodes_df['y'], errors='coerce')
        normal_mask = (y_train == 0)
        normal_nodes = (
            nodes_df.loc[normal_mask]
            .copy()
            .reset_index(drop=True)
        )
        if len(normal_nodes) == 0:
            raise ValueError("No normal nodes found.")
        # 僅使用 Normal-induced Subgraph
        edge_index = create_edge_index(
            normal_nodes,
            edges_df
        ).to(self.device)
        
        # 只使用 Normal Nodes 的 Features
        X_train = normal_nodes[features].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all():
            raise ValueError("Training features contain NaN or Inf.")
        X_scaled = self.scaler.fit_transform(X_train)
        x_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32,
            device=self.device
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model(x_tensor, edge_index)
            loss = criterion(
                reconstructed,
                x_tensor
            )
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}")
        return self

    def predict_score(self, nodes_df, edges_df, features):
        self.model.eval()
        edge_index = create_edge_index(nodes_df, edges_df).to(self.device)
        
        X_test = nodes_df[features].values
        if not np.isfinite(X_test).all():
            raise ValueError("Prediction features contain NaN or Inf.")
        X_scaled = self.scaler.transform(X_test)
        x_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(x_tensor, edge_index)
            mse_per_sample = torch.mean((reconstructed - x_tensor) ** 2, dim=1)
            
        scores = mse_per_sample.cpu().numpy()
        return scores

    def select_threshold(self, nodes_df, edges_df, features):
        """在 Validation 分別選擇 MSE 與 -MSE 的最佳 threshold"""
        mse_scores = self.predict_score(nodes_df, edges_df, features)
        
        y_series = pd.to_numeric(
            nodes_df["y"], errors="coerce"
        )
        valid_mask = y_series.notna()
        
        y_true = y_series[valid_mask].astype(int).to_numpy()
        if len(y_true) == 0 or len(np.unique(y_true)) != 2:
            raise ValueError(
                "Validation must contain both normal and illicit samples."
            )
        mse_scores = mse_scores[valid_mask.to_numpy()]

        def find_best_threshold(scores):
            thresholds = np.unique(scores)

            best_f1 = -1
            best_threshold = None

            for threshold in thresholds:
                y_pred = (scores >= threshold).astype(int)

                current_f1 = f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold

            return float(best_threshold), float(best_f1)
        
        # Original MSE
        self.threshold, self.original_best_f1 = (
            find_best_threshold(mse_scores)
        )

        # Reversed -MSE
        self.reverse_threshold, self.reverse_best_f1 = (
            find_best_threshold(-mse_scores)
        )

        print("\nValidation Threshold Selection")

        print(
            f"Original MSE threshold: {self.threshold:.6f}, "
            f"F1: {self.original_best_f1:.4f}"
        )

        print(
            f"Reversed -MSE threshold: {self.reverse_threshold:.6f}, "
            f"F1: {self.reverse_best_f1:.4f}"
        )

        return self

    def predict(self, nodes_df, edges_df, features):
        scores = self.predict_score(nodes_df, edges_df, features)
        return (scores >= self.threshold).astype(int)

def main(data_dir, output_dir, random_seed):
    # --- 5. 同步讀取 Node 與 Edge 的 CSV ---
    train = pd.read_csv(data_dir+'nodes_train.csv')
    val = pd.read_csv(data_dir+'nodes_validation.csv')
    test_before_event = pd.read_csv(data_dir+'nodes_test_pre_event.csv')
    test_after_event = pd.read_csv(data_dir+'nodes_test_post_event.csv')
    
    edge_train = pd.read_csv(data_dir+'edge_train.csv')
    edge_val = pd.read_csv(data_dir+'edge_validation.csv')
    edge_test_before = pd.read_csv(data_dir+'edge_test_pre_event.csv')
    edge_test_after = pd.read_csv(data_dir+'edge_test_post_event.csv')
    
    data_sets = [(train, 'train'), (val, 'val'), (test_before_event, 'test_before_event'), (test_after_event, 'test_after_event')]
    summarize_labels(data_sets, output_dir)
    ##########################################################################################################################
    features_name = [
        col for col in train.columns
        if col.startswith("feature_")
    ]

    input_dim = len(features_name)
    print(f"Number of input features: {input_dim}")

    model = GraphAutoencoderModel(
        input_dim=input_dim,
        latent_dim=64,
        epochs=50,
        random_seed=random_seed
    )
    
    # 訓練階段：傳入完整的 node 與 edge 進行圖卷積建立
    model.fit(train, edge_train, features_name)

    # 選擇閾值：傳入 Validation 的圖結構
    model.select_threshold(val, edge_val, features_name)
    ##########################################################################################################################
    
    output = Path(output_dir)
    metrics = {}
    plt.figure(figsize=(7, 6))
    has_curve = False
    
    # 將 edge 資料打包供迴圈呼叫
    test_packs = (
        ("val", val, edge_val), 
        ("test_before_event", test_before_event, edge_test_before), 
        ("test_after_event", test_after_event, edge_test_after)
    )
    
    for name, data, edges in test_packs:
        # 注意這裡傳入了 edges
        metrics[name], curve_data = evaluate_and_save(
            model, data, edges, features_name, name, output
        )
        if curve_data is not None:
            fpr, tpr = curve_data["original"]
            plt.plot(
                fpr,
                tpr,
                label=f"{name.upper()} Original "
                    f"(AUROC={metrics[name]['original']['auroc']:.4f})"
            )

            fpr, tpr = curve_data["reversed"]

            plt.plot(
                fpr,
                tpr,
                linestyle="--",
                label=f"{name.upper()} Reversed "
                    f"(AUROC={metrics[name]['reversed']['auroc']:.4f})"
            )

            has_curve = True
            
    if has_curve:
        plt.plot([0, 1], [0, 1], "--", label="Random (AUROC=0.5)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve: Validation / Test (GAE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "roc_curve.png", dpi=180)
    plt.close()

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, result in metrics.items():
        original_auc = result["original"].get("auroc")
        reversed_auc = result["reversed"].get("auroc")

        print(
            f"{name.upper()}: "
            f"Original AUROC={original_auc}, "
            f"Reversed AUROC={reversed_auc}"
        )
    print(f"Output: {output.resolve()}")
    return None

if __name__ == '__main__':
    random_seed = 42
    data_dir = f'./Data/'
    output_dir = f'./output/directed_GAE_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, random_seed)