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

# --- 新增 PyTorch Geometric 的 GCN 卷積層 ---
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
    edge_index = to_undirected(edge_index) # Undirected TODO
    return edge_index

def evaluate_and_save(model, data, edges, features_name, name, output):
    y_score = model.predict_score(data, edges, features_name)
    if model.threshold is None: raise ValueError("Threshold has not been selected.")
    if model.reverse_threshold is None: raise ValueError("Reversed threshold has not been selected.")
    original_score = y_score
    reversed_score = -y_score
    original_pred = (original_score >= model.threshold).astype(int)
    reversed_pred = (reversed_score >= model.reverse_threshold).astype(int)
    known = data["y"].notna()
    actual = data["y"]
    pd.DataFrame({"time_step": data["time_step"], "y_true": actual, "y_pred": original_pred, "anomaly_score": original_score}).to_csv(output / f"{name}_original_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_step": data["time_step"], "y_true": actual, "y_pred": reversed_pred, "anomaly_score": reversed_score}).to_csv(output / f"{name}_reversed_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {"n_rows": int(len(data)), "n_labeled": int(known.sum()), "original": {}, "reversed": {}}
    curve_data = None
    if known.any():
        y_true = actual.loc[known].astype(int).to_numpy()
        original_score_labeled = original_score[known.to_numpy()]
        reversed_score_labeled = reversed_score[known.to_numpy()]
        original_pred_labeled = original_pred[known.to_numpy()]
        reversed_pred_labeled = reversed_pred[known.to_numpy()]
        def calculate_metrics(y_true, y_pred, y_score):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            result = {
                "precision_anomaly": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "recall_anomaly": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "f1_anomaly": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": (float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) == 2 else 0.0),
                "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            }
            if len(np.unique(y_true)) == 2:
                result["auroc"] = float(roc_auc_score(y_true, y_score))
                result["pr_auc"] = float(average_precision_score(y_true, y_score))
            else:
                result["auroc"] = None
                result["pr_auc"] = None
            return result
        metrics["original"] = calculate_metrics(y_true, original_pred_labeled, original_score_labeled)
        metrics["reversed"] = calculate_metrics(y_true, reversed_pred_labeled, reversed_score_labeled)
        if len(np.unique(y_true)) == 2:
            fpr_original, tpr_original, thresholds_original = roc_curve(y_true, original_score_labeled, pos_label=1)
            fpr_reversed, tpr_reversed, thresholds_reversed = roc_curve(y_true, reversed_score_labeled, pos_label=1)
            pd.DataFrame({"fpr": fpr_original, "tpr": tpr_original, "threshold": thresholds_original}).to_csv(output / f"{name}_original_roc_points.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"fpr": fpr_reversed, "tpr": tpr_reversed, "threshold": thresholds_reversed}).to_csv(output / f"{name}_reversed_roc_points.csv", index=False, encoding="utf-8-sig")
            curve_data = {"original": (fpr_original, tpr_original), "reversed": (fpr_reversed, tpr_reversed)}
            print(f"\n{name.upper()}")
            print(f"Original L1: AUROC={metrics['original']['auroc']:.4f}, AP={metrics['original']['pr_auc']:.4f}, F1={metrics['original']['f1_anomaly']:.4f}")
            print(f"Reversed -L1: AUROC={metrics['reversed']['auroc']:.4f}, AP={metrics['reversed']['pr_auc']:.4f}, F1={metrics['reversed']['f1_anomaly']:.4f}")
    return metrics, curve_data

def summarize_labels(data_sets, output_dir):
    output_dir = Path(output_dir)
    all_summary = []
    for data, name in data_sets:
        y = (
            data["y"]
            if "y" in data.columns
            else pd.Series(float("nan"), index=data.index)
        )
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
    summary.to_excel(
        output_dir / f"label_summary.xlsx",
        index=False
    )
    return summary

class GAE_Model(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(GAE_Model, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x, edge_index):
        z = torch.relu(self.conv1(x, edge_index))
        z = torch.relu(self.conv2(z, edge_index))
        return self.decoder(z)

class GraphAutoencoderModel:
    def __init__(self, input_dim, latent_dim=64, epochs=50, lr=1e-3, random_seed=42):
        set_seed(random_seed)
        self.random_seed = random_seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.reverse_threshold = None
        self.best_f1 = None
        self.reverse_best_f1 = None
        self.best_ap = None
        self.reverse_best_ap = None

    def fit(self, train_nodes_df, train_edges_df, val_nodes_df, val_edges_df, features):
        self.model = GAE_Model(self.input_dim, self.latent_dim).to(self.device)

        # ---------------------------------------------------------------------
        # Training Setup (Pure Normal Subgraph)
        # ---------------------------------------------------------------------
        y_train = pd.to_numeric(train_nodes_df['y'], errors='coerce')
        normal_mask = (y_train == 0)
        normal_nodes = (train_nodes_df.loc[normal_mask].copy().reset_index(drop=True))
        if len(normal_nodes) == 0: 
            raise ValueError("No normal nodes found in training data.")
        
        train_edge_index = create_edge_index(normal_nodes, train_edges_df).to(self.device)
        X_train = normal_nodes[features].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all(): 
            raise ValueError("Training features contain NaN or Inf.")
        X_train_scaled = self.scaler.fit_transform(X_train)
        x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------------
        # Validation Setup (Pure Normal Subgraph for Early Stopping)
        # ---------------------------------------------------------------------
        y_val_series = pd.to_numeric(val_nodes_df['y'], errors='coerce')
        val_normal_mask = (y_val_series == 0)
        val_normal_nodes = (val_nodes_df.loc[val_normal_mask].copy().reset_index(drop=True))
        if len(val_normal_nodes) == 0:
            raise ValueError("No normal nodes found in validation data.")
            
        # 透過 create_edge_index 自動過濾出兩端皆為 Validation Normal nodes 的連線
        val_edge_index = create_edge_index(val_normal_nodes, val_edges_df).to(self.device)
        
        X_val = val_normal_nodes[features].to_numpy(dtype=np.float32)
        if not np.isfinite(X_val).all():
            raise ValueError("Validation features contain NaN or Inf.")
        X_val_scaled = self.scaler.transform(X_val)
        x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------------
        # Optimization & Early Stopping Loop
        # ---------------------------------------------------------------------
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()
        
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            reconstructed = self.model(x_train_tensor, train_edge_index)
            loss = criterion(reconstructed, x_train_tensor)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            self.model.eval()
            with torch.no_grad():
                # 這裡的 Validation loss 完全建構在 Normal-induced subgraph 上
                val_reconstructed = self.model(x_val_tensor, val_edge_index)
                val_loss = criterion(val_reconstructed, x_val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                import copy
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0: 
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}!")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def predict_score(self, nodes_df, edges_df, features):
        self.model.eval()
        edge_index = create_edge_index(nodes_df, edges_df).to(self.device)
        X_test = nodes_df[features].values
        if not np.isfinite(X_test).all(): raise ValueError("Prediction features contain NaN or Inf.")
        X_scaled = self.scaler.transform(X_test)
        x_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(x_tensor, edge_index)
            l1_per_sample = torch.mean(torch.abs(reconstructed - x_tensor), dim=1)
        scores = l1_per_sample.cpu().numpy()
        return scores

    def select_threshold(self, val_nodes_df, val_edges_df, features):
        mae_scores = self.predict_score(val_nodes_df, val_edges_df, features)
        y_series = pd.to_numeric(val_nodes_df["y"], errors="coerce")
        valid_mask = y_series.notna()
        y_true = y_series[valid_mask].astype(int).to_numpy()
        if len(y_true) == 0 or len(np.unique(y_true)) != 2: raise ValueError("Validation must contain both normal and illicit samples.")
        mae_scores = mae_scores[valid_mask.to_numpy()]
        def find_best_threshold(scores):
            thresholds = np.unique(scores)
            best_f1 = -1
            best_threshold = None
            for threshold in thresholds:
                y_pred = (scores >= threshold).astype(int)
                current_f1 = f1_score(y_true, y_pred, zero_division=0)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
            return float(best_threshold), float(best_f1)
        
        self.best_ap = average_precision_score(y_true, mae_scores)
        self.threshold, self.best_f1 = (find_best_threshold(mae_scores))
        
        reversed_scores = -mae_scores
        self.reverse_best_ap = average_precision_score(y_true, reversed_scores)
        self.reverse_threshold, self.reverse_best_f1 = (find_best_threshold(reversed_scores))
        
        print("\nValidation Threshold Selection")
        print("\nOriginal MAE:")
        print(f"AP:        {self.best_ap:.4f}")
        print(f"Threshold: {self.threshold:.6f}")
        print(f"F1:        {self.best_f1:.4f}")
        print("\nReversed -MAE:")
        print(f"AP:        {self.reverse_best_ap:.4f}")
        print(f"Threshold: {self.reverse_threshold:.6f}")
        print(f"F1:        {self.reverse_best_f1:.4f}")
        return self
    
    def predict(self, nodes_df, edges_df, features):
        scores = self.predict_score(nodes_df, edges_df, features)
        return (scores >= self.threshold).astype(int)

def main(data_dir, output_dir, random_seed):
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
    
    features_name = [col for col in train.columns if col.startswith("feature_")]
    input_dim = len(features_name)
    print(f"Number of input features: {input_dim}")

    val_labeled_mask = val["y"].notna()
    val_labeled_nodes = val.loc[val_labeled_mask].copy().reset_index(drop=True)
    
    param_grid = {
        'latent_dim': [32, 64],
        'epochs': [50, 100, 150, 200],
        'lr': [1e-3, 5e-4]
    }
    
    import itertools
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    
    best_overall_ap = -1
    best_params = None
    best_model = None
    
    print(f"Starting Grid Search over {len(combinations)} combinations...")
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing parameters: {params}")
        
        current_model = GraphAutoencoderModel(
            input_dim=input_dim, 
            latent_dim=params['latent_dim'], 
            epochs=params['epochs'], 
            lr=params['lr'], 
            random_seed=random_seed
        )
        
        # Validation setup for Early Stopping inside fit: we use the whole validation set to build the graph structure
        current_model.fit(train, edge_train, val, edge_val, features_name)
        
        val_scores = current_model.predict_score(val, edge_val, features_name)
        val_scores_labeled = val_scores[val_labeled_mask.to_numpy()]
        val_y_labeled = val.loc[val_labeled_mask, "y"].astype(int).to_numpy()

        val_ap = average_precision_score(val_y_labeled, -val_scores_labeled)
        print(f"Validation Reversed AP: {val_ap:.4f}")
        
        if val_ap > best_overall_ap:
            best_overall_ap = val_ap
            best_params = params
            import copy
            best_model = copy.deepcopy(current_model)

    best_model.select_threshold(val, edge_val, features_name)

    print("\n==================================================")
    print("Grid Search Completed!")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Reversed AP: {best_overall_ap:.4f}")
    print(f"Best Validation Reversed F1: {best_model.reverse_best_f1:.4f}")
    print(f"Best Reversed Threshold: {best_model.reverse_threshold:.6f}")
    
    output = Path(output_dir)
    metrics = {}
    plt.figure(figsize=(7, 6))
    has_curve = False
    
    test_packs = (("val", val, edge_val), ("test_before_event", test_before_event, edge_test_before), ("test_after_event", test_after_event, edge_test_after))
    for name, data, edges in test_packs:
        metrics[name], curve_data = evaluate_and_save(best_model, data, edges, features_name, name, output)
        if curve_data is not None:
            fpr, tpr = curve_data["original"]
            plt.plot(fpr, tpr, label=f"{name.upper()} Original (AUROC={metrics[name]['original']['auroc']:.4f})")
            fpr, tpr = curve_data["reversed"]
            plt.plot(fpr, tpr, linestyle="--", label=f"{name.upper()} Reversed (AUROC={metrics[name]['reversed']['auroc']:.4f})")
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
    
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, result in metrics.items():
        original_auc = result["original"].get("auroc")
        reversed_auc = result["reversed"].get("auroc")
        print(f"{name.upper()}: Original AUROC={original_auc}, Reversed AUROC={reversed_auc}")
    print(f"Output: {output.resolve()}")
    return None

if __name__ == '__main__':
    random_seed = 42
    data_dir = f'./Data/'
    output_dir = f'./output/GAE_best_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, random_seed)