"""Random forest training and experiment orchestration."""

import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.eval import evaluate_and_save, save_roc_plot
from src.utils.summarize import summarize_labels
from src.utils.artifacts import write_artifact_index


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
    labeled = train[label].notna()
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
        model = RandomForestClassifier(**config["model"]["params"], random_state=seed)
        model.fit(train.loc[labeled, features], train.loc[labeled, label].astype(int))
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
        importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        importance.insert(0, "rank", importance.index + 1)
        importance.to_csv(dirs["eval"] / f"{prefix}_feature_importance.csv",
                          index=False, encoding="utf-8-sig")
        write_artifact_index(
            output, seed, model_name, set_name, features,
            index_path=config["model_index"],
            input_features=settings.get("encoder_features"),
            embedding_columns=settings.get("embedding_columns"),
            feature_source=settings.get("feature_source"),
            **settings.get("encoder_artifacts", {}),
        )
        all_metrics[seed] = metrics
        print(f"set={set_name}, seed={seed}: labeled train rows={int(labeled.sum())}, features={len(features)}")
        for name, result in metrics.items():
            print(f"  {name}: labeled={result['n_labeled']}, AUROC={result['auroc']}")
    print(f"Output: {output}")
    return all_metrics
