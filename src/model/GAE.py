"""Graph Autoencoder (GAE) Baseline and experiment orchestration."""

import json
import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# PyTorch Geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_edge_index(nodes_df, edges_df, undirected=True):
    """根據節點與邊清單建立 edge_index，並支援單向或雙向圖設定。"""
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
    
    if undirected:
        edge_index = to_undirected(edge_index)
        
    return edge_index


class GAE_Encoder_Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(GAE_Encoder_Decoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, edge_index):
        z = torch.relu(self.conv1(x, edge_index))
        z = torch.relu(self.conv2(z, edge_index))
        return self.decoder(z)


class GraphAutoencoderModel:
    def __init__(self, input_dim, latent_dim=64, epochs=100, lr=5e-4, 
                 loss_type="L1", undirected=True, reverse_score=False, random_seed=42):
        set_seed(random_seed)
        self.random_seed = random_seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.loss_type = loss_type.upper()
        self.undirected = undirected
        self.reverse_score = reverse_score
        
        self.model = GAE_Encoder_Decoder(input_dim, latent_dim).to(self.device)
        self.scaler = StandardScaler()
        
        self.threshold = None
        self.best_f1 = None
        self.classes_ = np.array([0, 1])

    def fit(self, train_nodes_df, train_edges_df, features):
        """
        防漏規範：只使用純正常節點 (y==0) 建立 Normal-induced Subgraph 與進行訓練
        """
        y_train = pd.to_numeric(train_nodes_df['y'], errors='coerce')
        normal_mask = (y_train == 0)
        normal_nodes = train_nodes_df.loc[normal_mask].copy().reset_index(drop=True)
        
        if len(normal_nodes) == 0:
            raise ValueError("No normal nodes found in training data.")
            
        edge_index = create_edge_index(normal_nodes, train_edges_df, undirected=self.undirected).to(self.device)
        
        X_train = normal_nodes[features].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all():
            raise ValueError("Training features contain NaN or Inf.")
            
        X_scaled = self.scaler.fit_transform(X_train)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.loss_type == "MSE":
            criterion = nn.MSELoss()
        elif self.loss_type == "L1":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Use 'MSE' or 'L1'.")
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model(x_tensor, edge_index)
            loss = criterion(reconstructed, x_tensor)
            loss.backward()
            optimizer.step()
            
        return self

    def predict_score(self, nodes_df, edges_df, features):
        """計算圖重構誤差，並支援分數倒置。"""
        self.model.eval()
        edge_index = create_edge_index(nodes_df, edges_df, undirected=self.undirected).to(self.device)
        
        X_test = nodes_df[features].values
        if not np.isfinite(X_test).all():
            raise ValueError("Prediction features contain NaN or Inf.")
            
        X_scaled = self.scaler.transform(X_test)
        x_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(x_tensor, edge_index)
            if self.loss_type == "MSE":
                error_per_sample = torch.mean((reconstructed - x_tensor) ** 2, dim=1)
            else:
                error_per_sample = torch.mean(torch.abs(reconstructed - x_tensor), dim=1)
            
        scores = error_per_sample.cpu().numpy()
        
        if self.reverse_score:
            scores = -scores
            
        return scores

    def select_threshold(self, val_nodes_df, val_edges_df, features):
        """在 Validation 選擇最佳 F1 threshold。"""
        scores = self.predict_score(val_nodes_df, val_edges_df, features)
        y_series = pd.to_numeric(val_nodes_df["y"], errors="coerce")
        valid_mask = y_series.notna()
        
        y_true = y_series[valid_mask].astype(int).to_numpy()
        if len(y_true) == 0 or len(np.unique(y_true)) != 2:
            raise ValueError("Validation must contain both normal and illicit samples.")
            
        scores = scores[valid_mask.to_numpy()]

        thresholds = np.unique(scores)
        best_f1 = -1
        best_threshold = None

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        self.best_f1 = float(best_f1)
        return self

    def predict(self, nodes_df, edges_df, features):
        # 這裡會由外部傳入 edges，因此我們覆寫 predict 以適應圖模型架構
        raise NotImplementedError("Use predict_score or evaluate directly with graph edges.")

    def predict_proba(self, nodes_df, edges_df, features):
        """相容 evaluate_and_save 的機率呼叫"""
        anomaly_scores = self.predict_score(nodes_df, edges_df, features)
        prob_anomaly = 1 / (1 + np.exp(-anomaly_scores))
        return np.column_stack([1 - prob_anomaly, prob_anomaly])


def run(config):
    settings = config["data"]
    files = {"train": settings["train_file"], "val": settings["val_file"],
             **settings["test_files"]}
    datasets = {name: pd.read_csv(config["input"] / filename)
                for name, filename in files.items()}
                
    # 同步讀取邊資料 (Edge Files)
    edge_files = settings.get("edge_files", {})
    edges_datasets = {name: pd.read_csv(config["input"] / filename)
                      for name, filename in edge_files.items()}
                      
    results = {}
    for feature_set in settings["feature_sets"]:
        set_settings = {**settings, **feature_set}
        results[feature_set["set_name"]] = _run_feature_set(config, datasets, edges_datasets, set_settings)
    return results


def _run_feature_set(config, datasets, edges_datasets, settings):
    train = datasets["train"]
    val = datasets["val"]
    label = settings.get("label_column", "y")
    
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    
    features = settings.get("feature_columns")
    if features is None:
        features = list(train.columns)[settings.get("feature_start_index", 4):]

    if not isinstance(features, list) or not features or len(features) != len(set(features)) or label in features:
        raise ValueError("Features must be a nonempty unique list excluding the label column")
        
    for name, data in datasets.items():
        missing = set(features) - set(data.columns)
        if missing:
            raise ValueError(f"{name}: missing features {sorted(missing)}")
        if label in data and not data[label].dropna().isin([0, 1]).all():
            raise ValueError(f"{name}: labels must be 0, 1, or missing")

    val_labeled = val[label].notna()
    
    # -------------------------------------------------------------------
    # 動態輸出資料夾命名區隔 (Model + Loss + Graph Type + Reverse)
    # -------------------------------------------------------------------
    loss_type = model_params.get('loss_type', 'L1')
    graph_type = "undirected" if model_params.get('undirected', True) else "directed"
    is_reverse = model_params.get('reverse_score', False)
    
    rev_tag = "_reverse" if is_reverse else ""
    set_name = settings.get("set_name", "basic")
    
    output = config["output"] / f"{model_name}_{loss_type}_{graph_type}{rev_tag}_output" / set_name
    dirs = {name: output / name for name in ("model", "eval", "predict")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    all_metrics = {}
    
    for seed in config["random_seeds"]:
        set_seed(seed)
        prefix = f"{set_name}_seed_{seed}_{model_name}"
        
        if model_name == "GAE":
            input_dim = len(features)
            model = GraphAutoencoderModel(
                input_dim=input_dim,
                latent_dim=model_params.get("latent_dim", 64),
                epochs=model_params.get("epochs", 100),
                lr=model_params.get("lr", 5e-4),
                loss_type=loss_type,
                undirected=model_params.get("undirected", True),
                reverse_score=is_reverse,
                random_seed=seed
            )
            
            # 訓練階段：傳入訓練集與對應的邊檔案
            model.fit(train, edges_datasets["train"], features)
            
            # 閾值選擇：傳入驗證集與對應的邊檔案
            model.select_threshold(val, edges_datasets["val"], features)
        else:
            raise ValueError(f"Unknown model name: {model_name}. Use GAE.")

        joblib.dump({"model": model, "features": features, "seed": seed,
                     "label_column": label, "set_name": set_name},
                    dirs["model"] / f"{prefix}_model.joblib")
                    
        summarize_labels([(data, name) for name, data in datasets.items()],
                         dirs["eval"] / f"{prefix}_label_summary.xlsx", label)
                         
        metrics, curves = {}, {}
        
        # 測試評估迴圈（將對應的 edge 檔案餵進去）
        test_packs = {
            "val": (val, edges_datasets["val"]),
            "test_before_event": (datasets["test_before_event"], edges_datasets["test_before_event"]),
            "test_after_event": (datasets["test_after_event"], edges_datasets["test_after_event"])
        }
        
        for name, (data, edges) in test_packs.items():
            # 因為 GAE 需要用到圖結構，我們透過包裝類別呼叫評估
            # 建立一個臨時具備 predict 與 predict_proba 的輕量介面供 evaluate_and_save 使用
            class GAE_EvaluatorWrapper:
                def __init__(self, m, d, f):
                    self.m = m
                    self.edges = d
                    self.features = f
                    self.classes_ = m.classes_
                def predict(self, x):
                    # 傳入當前資料集的 edges
                    scores = self.m.predict_score(data, self.edges, self.features)
                    return (scores >= self.m.threshold).astype(int)
                def predict_proba(self, x):
                    return self.m.predict_proba(data, self.edges, self.features)

            eval_model = GAE_EvaluatorWrapper(model, edges, features)
            
            metrics[name], curve = evaluate_and_save(
                eval_model, data, features, name, dirs["eval"], dirs["predict"], prefix, label)
            if curve is not None:
                curves[name] = curve
                
        save_roc_plot(curves, metrics, dirs["eval"] / f"{prefix}_roc_curve.png")
        
        (dirs["eval"] / f"{prefix}_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            
        write_artifact_index(
            output, seed, model_name, set_name, features,
            index_path=config["model_index"],
            input_features=settings.get("encoder_features"),
            embedding_columns=settings.get("embedding_columns"),
            feature_source=settings.get("feature_source"),
            **settings.get("encoder_artifacts", {}),
        )
        
        all_metrics[seed] = metrics
        
        print(f"set={set_name}, seed={seed}: features={len(features)}, graph={graph_type}, loss={loss_type}")
        print(f"  Validation best F1: {model.best_f1:.4f}, Threshold: {model.threshold:.6f}")
        for name, result in metrics.items():
            print(f"  {name}: labeled={result['n_labeled']}, AUROC={result.get('auroc', 'N/A')}")
            
    print(f"Output: {output}")
    return all_metrics