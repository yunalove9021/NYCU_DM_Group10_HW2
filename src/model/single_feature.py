"""Single Feature Baseline and experiment orchestration."""

import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index


class SingleFeatureModel:
    def __init__(self, feature_name="feature_5"):
        self.feature_name = feature_name
        self.threshold = None
        self.direction = None
        self.best_f1 = None
        # 為了相容通用評估函數，增加 classes_
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """使用 Validation 決定 score direction 和最佳 threshold。"""
        scores = X[self.feature_name].to_numpy()
        y = np.asarray(y).astype(int)

        auc_positive = roc_auc_score(y, scores)
        auc_negative = roc_auc_score(y, -scores)

        if auc_positive >= auc_negative:
            self.direction = 1
            anomaly_scores = scores
        else:
            self.direction = -1
            anomaly_scores = -scores

        thresholds = np.unique(anomaly_scores)
        best_f1 = -1
        best_threshold = None

        from sklearn.metrics import f1_score
        for threshold in thresholds:
            y_pred = (anomaly_scores >= threshold).astype(int)
            current_f1 = f1_score(y, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        self.best_f1 = float(best_f1)
        return self

    def predict_score(self, X):
        scores = X[self.feature_name].to_numpy()
        return self.direction * scores

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
    
    # 針對 single_feature，我們不需要跑完所有的 feature_sets，只跑預設的 basic 就好
    if config["model"]["name"] == "single_feature":
        # 尋找 basic 的 feature set，或者直接只跑第一組
        target_set = settings["feature_sets"][0]
        for fset in settings["feature_sets"]:
            if fset.get("set_name") == "basic":
                target_set = fset
                break
        set_settings = {**settings, **target_set}
        results[target_set["set_name"]] = _run_feature_set(config, datasets, set_settings)
    else:
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
    
    # 關鍵修改：如果是 single_feature，強行抽換 features 列表
    if model_name == "single_feature":
        target_feature = model_params.get("feature_name", "feature_5")
        features = [target_feature]
    else:
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

    val_labeled = val[label].notna()
    
    set_name = settings.get("set_name", "basic")
    output = config["output"] / f"{model_name}_{target_feature}_output" / set_name
    dirs = {name: output / name for name in ("model", "eval", "predict")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    all_metrics = {}
    
    # Single Feature 其實沒有隨機性，所以只跑第一個 seed 即可，避免重複計算
    seeds_to_run = [config["random_seeds"][0]] if model_name == "single_feature" else config["random_seeds"]
    
    for seed in seeds_to_run:
        random.seed(seed)
        np.random.seed(seed)
        prefix = f"{set_name}_seed_{seed}_{model_name}"
        
        # 實例化與訓練
        if model_name == "single_feature":
            model = SingleFeatureModel(feature_name=target_feature)
            # 防漏規範：Single Feature 只使用 Validation 找方向，完全不碰 train
            model.fit(
                val.loc[val_labeled, features],
                val.loc[val_labeled, label]
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

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
        
        print(f"set={set_name}, Validation best F1={model.best_f1:.4f}, direction={'Higher=Anomaly' if model.direction==1 else 'Lower=Anomaly'}, feature={target_feature}")
        for name, result in metrics.items():
            print(f"  {name}: labeled={result['n_labeled']}, AUROC={result.get('auroc', 'N/A')}")
            
    print(f"Output: {output}")
    return all_metrics