"""Train a supervised GCN, concatenate node embeddings, then fit a random forest."""

import json

import joblib
import pandas as pd
import torch

from src.encoder.gcn import encode_nodes, fit_encoder
from src.model.RF import _run_feature_set
from src.utils.graph import edge_indices


def run(config):
    settings = config["data"]
    files = {"train": settings["train_file"], "val": settings["val_file"],
             **settings["test_files"]}
    datasets = {name: pd.read_csv(config["input"] / filename) for name, filename in files.items()}
    edges = {name: pd.read_csv(config["input"] / filename)
             for name, filename in settings["edge_files"].items()}
    encoder_settings = config["model"]["encoder"]
    label = settings.get("label_column", "y")
    for name, data in datasets.items():
        edge_indices(data, edges[name], encoder_settings.get("undirected", False))
        if label in data and not data[label].dropna().isin([0, 1]).all():
            raise ValueError(f"{name}: labels must be binary or missing")
    embedding_names = [f"emb_dim_{i}" for i in range(1, 101)]
    results = {}
    for feature_set in settings["feature_sets"]:
        set_name = feature_set["set_name"]
        features = feature_set["feature_columns"]
        if features is None:
            features = list(datasets["train"].columns)[settings.get("feature_start_index", 4):]
        if not features or label in features:
            raise ValueError("GCN input features must be nonempty and exclude the label")
        for name, data in datasets.items():
            missing = set(features) - set(data.columns)
            if missing:
                raise ValueError(f"{name}: missing features {sorted(missing)}")
            if set(embedding_names) & set(data.columns):
                raise ValueError("Input already contains emb_dim_1..100; avoid overwriting embeddings")
        output = config["output"] / "GCN_RF_output" / set_name
        dirs = {name: output / name for name in ("model", "eval", "predict")}
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        results[set_name] = {}
        for seed in config["random_seeds"]:
            prefix = f"{set_name}_seed_{seed}_GCN_RF"
            encoder, scaler, report = fit_encoder(
                datasets, edges, features, label, encoder_settings, seed)
            torch.save({"state_dict": {key: value.cpu() for key, value in encoder.state_dict().items()},
                        "input_features": features, "embedding_columns": embedding_names,
                        "settings": encoder_settings, "seed": seed, "set_name": set_name,
                        "label_column": label}, dirs["model"] / f"{prefix}_encoder.pt")
            joblib.dump(scaler, dirs["model"] / f"{prefix}_scaler.joblib")
            (dirs["eval"] / f"{prefix}_gcn_training.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            augmented = {}
            for name, data in datasets.items():
                values = encode_nodes(encoder, scaler, data, edges[name], features,
                                      encoder_settings.get("undirected", False))
                embeddings = pd.DataFrame(values, columns=embedding_names, index=data.index)
                identifiers = data[[col for col in ("txId", "time_step") if col in data]].copy()
                pd.concat([identifiers, embeddings], axis=1).to_csv(
                    dirs["predict"] / f"{prefix}_{name}_embeddings.csv", index=False, encoding="utf-8-sig")
                augmented[name] = pd.concat([data, embeddings], axis=1)
            seed_config = {**config, "random_seeds": [seed]}
            rf_settings = {**settings, "set_name": set_name,
                           "feature_columns": features + embedding_names,
                           "feature_source": feature_set.get("feature_source"),
                           "encoder_features": features, "embedding_columns": embedding_names,
                           "encoder_artifacts": {
                               "encoder": dirs["model"] / f"{prefix}_encoder.pt",
                               "scaler": dirs["model"] / f"{prefix}_scaler.joblib",
                               "embeddings": {name: dirs["predict"] / f"{prefix}_{name}_embeddings.csv"
                                              for name in datasets},
                           }}
            results[set_name].update(_run_feature_set(seed_config, augmented, rf_settings))
    return results
