"""Autoencoder Baseline and experiment orchestration."""

import json
import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class PytorchAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(PytorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel:
    def __init__(self, input_dim, latent_dim=32, epochs=50, batch_size=256, lr=5e-4, 
                 loss_type="L1", reverse_score=False, random_seed=42):
        set_seed(random_seed)
        self.random_seed = random_seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_type = loss_type.upper()
        self.reverse_score = reverse_score
        
        self.model = PytorchAutoencoder(input_dim, latent_dim).to(self.device)
        self.scaler = StandardScaler()
        
        self.threshold = None
        self.best_f1 = None
        self.classes_ = np.array([0, 1])

    def fit(self, X_train):
        """使用 Training 的純正常資料進行無監督訓練。"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        tensor_x = torch.FloatTensor(X_train_scaled)
        dataset = TensorDataset(tensor_x)
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.loss_type == "MSE":
            criterion = nn.MSELoss()
        elif self.loss_type == "L1":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Use 'MSE' or 'L1'.")
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        return self

    def predict_score(self, X):
        """計算重建誤差（MSE 或 L1）。若 reverse_score=True 則取負號調整。"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            if self.loss_type == "MSE":
                error_per_sample = torch.mean((reconstructed - tensor_x) ** 2, dim=1)
            else:
                error_per_sample = torch.mean(torch.abs(reconstructed - tensor_x), dim=1)
            
        scores = error_per_sample.cpu().numpy()
        
        if self.reverse_score:
            scores = -scores
            
        return scores

    def select_threshold(self, X_val, y_val):
        """使用 Validation 選擇最佳 F1 threshold。"""
        scores = self.predict_score(X_val)
        y_val = np.asarray(y_val).astype(int)

        if len(y_val) == 0:
            raise ValueError("Validation has no labeled samples.")

        if len(np.unique(y_val)) != 2:
            raise ValueError("Validation must contain both normal and illicit samples.")

        thresholds = np.unique(scores)
        best_f1 = -1
        best_threshold = None

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            current_f1 = f1_score(y_val, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        self.best_f1 = float(best_f1)
        return self

    def predict(self, X):
        scores = self.predict_score(X)
        return (scores >= self.threshold).astype(int)

    def predict_proba(self, X):
        """相容 evaluate_and_save 的機率呼叫"""
        anomaly_scores = self.predict_score(X)
        prob_anomaly = 1 / (1 + np.exp(-anomaly_scores))
        return np.column_stack([1 - prob_anomaly, prob_anomaly])


def run(config):
    settings = config["data"]
    files = {"train": settings["train_file"], "val": settings["val_file"],
             **settings["test_files"]}
    datasets = {name: pd.read_csv(config["input"] / filename)
                for name, filename in files.items()}
    results = {}
    
    for feature_set in settings["feature_sets"]:
        set_settings = {**settings, **feature_set}
        results[feature_set["set_name"]] = _run_feature_set(config, datasets, set_settings)
            
    return results


def _run_feature_set(config, datasets, settings):
    train = datasets["train"]
    val = datasets["val"]
    label = settings.get("label_column", "y")
    
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    loss_type = model_params.get('loss_type', 'MSE')
    is_reverse = model_params.get('reverse_score', False)
    
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
            
    if label not in train or not train[label].notna().any():
        raise ValueError("Training data must contain labeled rows")

    labeled = train[label].notna()
    val_labeled = val[label].notna()
    
    # 防漏機制：僅擷取訓練集中的純正常節點
    normal_mask = (train[label] == 0)
    train_normal_features = train.loc[labeled & normal_mask, features]
    
    set_name = settings.get("set_name", "basic")
    if is_reverse:
        output = config["output"] / f"{model_name}_{loss_type}_reverse_output" / set_name
    else:
        output = config["output"] / f"{model_name}_{loss_type}_output" / set_name
    dirs = {name: output / name for name in ("model", "eval", "predict")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    all_metrics = {}
    
    for seed in config["random_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        prefix = f"{set_name}_seed_{seed}_{model_name}"
        
        if model_name == "AE":
            input_dim = len(features)
            model = AutoencoderModel(
                input_dim=input_dim,
                latent_dim=model_params.get("latent_dim", 32),
                epochs=model_params.get("epochs", 50),
                batch_size=model_params.get("batch_size", 256),
                lr=model_params.get("lr", 5e-4),
                loss_type=model_params.get("loss_type", "L1"),
                reverse_score=model_params.get("reverse_score", False),
                random_seed=seed
            )
            
            # 訓練與閾值選取
            model.fit(train_normal_features)
            model.select_threshold(
                val.loc[val_labeled, features],
                val.loc[val_labeled, label]
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}. Use AE.")

        joblib.dump({"model": model, "features": features, "seed": seed,
                     "label_column": label, "set_name": set_name},
                    dirs["model"] / f"{prefix}_model.joblib")
                    
        summarize_labels([(data, name) for name, data in datasets.items()],
                         dirs["eval"] / f"{prefix}_label_summary.xlsx", label)
                         
        metrics, curves = {}, {}
        for name, data in datasets.items():
            if name == "train":
                continue
            metrics[name], curve = evaluate_and_save(
                model, data, features, name, dirs["eval"], dirs["predict"], prefix, label)
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
        
        print(f"set={set_name}, seed={seed}: pure normal train rows={len(train_normal_features)}, features={len(features)}")
        print(f"  Validation best F1: {model.best_f1:.4f}, Threshold: {model.threshold:.6f}")
        for name, result in metrics.items():
            print(f"  {name}: labeled={result['n_labeled']}, AUROC={result.get('auroc', 'N/A')}")
            
    print(f"Output: {output}")
    return all_metrics