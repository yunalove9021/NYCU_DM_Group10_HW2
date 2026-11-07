import json
import random

import joblib
import numpy as np
import pandas as pd

# 引入專案架構內的通用 utils 模組
from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index

class RandomScoreModel:
    """A baseline model that assigns a uniform random anomaly score."""
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        # 隨機模型不需要真實訓練，但為了 API 相容性保留 fit 函數
        return self

    def predict_proba(self, X):
        anomaly_score = self.rng.uniform(0, 1, size=len(X))
        normal_score = 1 - anomaly_score
        return np.column_stack([normal_score, anomaly_score])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

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
    """Train every seed for one feature set using the already loaded data."""
    train = datasets["train"]
    label = settings.get("label_column", "y")
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

    # -------------------------------------------------------------------
    # 防漏機制 (Data Leakage Prevention)
    # 對於無監督的 Baseline 模型，僅萃取 Training Set 中的純正常節點 (y == 0)
    # -------------------------------------------------------------------
    labeled_mask = train[label].notna()
    normal_mask = (train[label] == 0)
    # 嚴格的訓練特徵：只有已標記且為正常的節點
    train_normal_features = train.loc[labeled_mask & normal_mask, features]
    
    model_name = config["model"]["name"]
    set_name = settings.get("set_name", "basic")
    output = config["output"] / f"{model_name}_output" / set_name
    dirs = {name: output / name for name in ("model", "eval", "predict")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    all_metrics = {}
    for seed in config["random_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        prefix = f"{set_name}_seed_{seed}_{model_name}"
        
        # 實例化模型
        model_params = config["model"].get("params", {})
        model_seed = model_params.get("random_seed", seed)
        model = RandomScoreModel(random_seed=model_seed)
        
        # 使用防漏後的純正常節點進行擬合
        model.fit(train_normal_features)
        
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
        
        # 為了報告清晰，我們把純正常節點的數量印出來
        pure_normal_count = len(train_normal_features)
        print(f"set={set_name}, seed={seed}: pure normal train rows={pure_normal_count}, features={len(features)}")
        for name, result in metrics.items():
            print(f"  {name}: labeled={result['n_labeled']}, AUROC={result.get('auroc', 'N/A')}")
            
    print(f"Output: {output}")
    return all_metrics