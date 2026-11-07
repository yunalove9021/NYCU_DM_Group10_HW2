"""Isolation Forest Baseline and experiment orchestration."""

import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index


class IsolationForestModel:
    def __init__(self, n_estimators=300, random_seed=42, reverse_score=False):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_seed,
            n_jobs=-1
        )
        self.reverse_score = reverse_score
        self.threshold = None
        self.best_f1 = None
        self.classes_ = np.array([0, 1])

    def fit(self, X_train):
        """使用 Training 的純正常資料進行無監督訓練。"""
        self.model.fit(X_train)
        return self

    def predict_score(self, X):
        """
        Isolation Forest 預設 decision_function 越低越異常。
        常規作法是加上負號，讓分數越高越異常。
        但若設定 reverse_score=True，則取消負號，適應 Elliptic 資料集洗錢節點極度密集的特性。
        """
        scores = -self.model.decision_function(X)
        if self.reverse_score:
            scores = -scores  # 再次反轉，負負得正
        return scores

    def select_threshold(self, X_val, y_val):
        """使用 Validation 選擇最佳 F1 threshold。"""
        scores = self.predict_score(X_val)
        y_val = np.asarray(y_val).astype(int)

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
        # 用 sigmoid 將無界分數壓到 0-1 區間作為近似機率
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
    is_reverse = model_params.get("reverse_score", False)
    
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
    
    # -------------------------------------------------------------------
    # 防漏機制 (Data Leakage Prevention)
    # 針對 IF 無監督模型，僅萃取 Training Set 中的純正常節點
    # -------------------------------------------------------------------
    normal_mask = (train[label] == 0)
    train_normal_features = train.loc[labeled & normal_mask, features]
    
    set_name = settings.get("set_name", "basic")
    
    if is_reverse:
        output = config["output"] / f"{model_name}_reverse_output" / set_name
    else:
        output = config["output"] / f"{model_name}_output" / set_name
    dirs = {name: output / name for name in ("model", "eval", "predict")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    all_metrics = {}
    
    for seed in config["random_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        prefix = f"{set_name}_seed_{seed}_{model_name}"
        
        # 實例化與訓練 IF 模型
        if model_name == "isolation_forest":
            model = IsolationForestModel(**model_params, random_seed=seed)
            
            # 使用純正常節點訓練
            model.fit(train_normal_features)
            
            # 使用驗證集找最佳 F1 Threshold
            model.select_threshold(
                val.loc[val_labeled, features],
                val.loc[val_labeled, label]
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}. Use IF.")

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