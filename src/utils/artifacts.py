"""Maintain configs/model.json and load one indexed model/encoder bundle."""

import json
import os
from pathlib import Path

import joblib


def write_artifact_index(output, seed, model_name, set_name, features, *, index_path,
                            input_features=None, embedding_columns=None,
                            feature_source=None, encoder=None, scaler=None,
                            embeddings=None):
    output = Path(output).resolve()
    index_path = Path(index_path).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{set_name}_seed_{seed}_{model_name}"
    feature_file = output / "model" / f"{prefix}_features.json"
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    feature_file.write_text(json.dumps({
        "input_features": features if input_features is None else input_features,
        "embedding_columns": embedding_columns or [],
        "model_features": features,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def relative(path):
        return Path(os.path.relpath(Path(path).resolve(), index_path.parent)).as_posix() if path else None

    manifest = {
        "schema_version": 1, "seed": seed, "model_name": model_name,
        "set_name": set_name,
        "path": {
            "model": relative(output / "model" / f"{prefix}_model.joblib"),
            "feature_table": relative(feature_file),
            "feature_source": relative(feature_source),
            "encoder": relative(encoder), "scaler": relative(scaler),
            "embeddings": {name: relative(path) for name, path in (embeddings or {}).items()},
        },
    }
    entries = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    if not isinstance(entries, dict):
        raise ValueError("Model index must be a JSON object")
    key = f"seed_{seed}_{model_name}_{set_name}"
    entries[key] = manifest
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return key


def load_manifest(path, index):
    """Select one index and resolve paths relative to configs/model.json."""
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if index not in entries:
        raise KeyError(f"Model index '{index}' not found in {path}")
    manifest = entries[index]
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest schema_version")
    paths = manifest["path"]

    def resolve(value, required=True):
        if value is None:
            return None
        target = (path.parent / value).resolve()
        if required and not target.is_file():
            raise FileNotFoundError(target)
        return target

    resolved = {key: resolve(paths[key]) for key in ("model", "feature_table", "encoder", "scaler")}
    if resolved["model"] is None or resolved["feature_table"] is None:
        raise ValueError("Manifest requires model and feature_table paths")
    if (resolved["encoder"] is None) != (resolved["scaler"] is None):
        raise ValueError("Encoder and scaler must be supplied together")
    resolved["feature_source"] = resolve(paths.get("feature_source"), required=False)
    # Training embedding exports are optional when predicting on a new CSV.
    resolved["embeddings"] = {name: resolve(value, required=False) for name, value in paths.get("embeddings", {}).items()}
    return {**manifest, "index": index, "path": resolved}


def load_artifacts(path, index, device="cpu"):
    """Load classifier, feature order, and optional GCN without retraining."""
    manifest = load_manifest(path, index)
    paths = manifest["path"]
    model_artifact = joblib.load(paths["model"])
    features = json.loads(paths["feature_table"].read_text(encoding="utf-8"))
    if model_artifact["features"] != features["model_features"]:
        raise ValueError("Feature table does not match model feature order")
    if model_artifact["seed"] != manifest["seed"] or model_artifact["set_name"] != manifest["set_name"]:
        raise ValueError("Model seed/set_name does not match manifest")
    bundle = {"manifest": manifest, "model": model_artifact["model"],
              "features": features, "encoder": None, "scaler": None}
    if paths["encoder"] is not None:
        from src.encoder.gcn import load_encoder
        encoder, scaler, checkpoint = load_encoder(paths["encoder"], paths["scaler"], device)
        if (checkpoint["seed"] != manifest["seed"] or checkpoint["set_name"] != manifest["set_name"]
                or checkpoint["input_features"] != features["input_features"]
                or checkpoint["embedding_columns"] != features["embedding_columns"]):
            raise ValueError("Encoder does not match manifest or feature table")
        bundle.update(encoder=encoder, scaler=scaler, encoder_settings=checkpoint["settings"])
    return bundle


def prepare_model_input(bundle, nodes, edges=None):
    """Build ordered classifier input, generating embeddings when needed."""
    data = nodes.copy()
    if bundle["encoder"] is not None:
        if edges is None:
            raise ValueError("GCN prediction requires edges")
        from src.encoder.gcn import encode_nodes
        import pandas as pd
        values = encode_nodes(bundle["encoder"], bundle["scaler"], data, edges,
                              bundle["features"]["input_features"],
                              bundle["encoder_settings"].get("undirected", False))
        columns = bundle["features"]["embedding_columns"]
        data = pd.concat([data.drop(columns=columns, errors="ignore"),
                          pd.DataFrame(values, columns=columns, index=data.index)], axis=1)
    return data[bundle["features"]["model_features"]]
