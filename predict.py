"""Predict from an indexed RF or GCN_RF model without training."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.model.registry import MODEL_REGISTRY, validate_model_name
from src.utils.artifacts import load_artifacts, load_manifest, prepare_model_input
from src.utils.graph import edge_indices


ROOT = Path(__file__).resolve().parent


def predict(config_path, model_type, index, input_path, edge_path=None,
            output_dir=None, device="cpu"):
    """Validate input and return the path to a prediction CSV."""
    validate_model_name(model_type)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", index):
        raise ValueError("Model index may contain only letters, digits, underscores, or hyphens")
    manifest = load_manifest(config_path, index)
    if manifest["model_name"] != model_type:
        raise ValueError(f"Model type mismatch: requested {model_type}, index '{index}' is {manifest['model_name']}")
    if model_type == "GCN_RF" and edge_path is None:
        raise ValueError("GCN_RF requires --edge pointing to an edge CSV")
    if model_type == "RF" and edge_path is not None:
        raise ValueError("RF does not use --edge; select GCN_RF for graph prediction")
    has_encoder = manifest["path"]["encoder"] is not None
    if has_encoder != (model_type == "GCN_RF"):
        raise ValueError(f"Encoder configuration does not match model type {model_type}")
    input_path = Path(input_path)
    nodes = pd.read_csv(input_path)
    feature_table = json.loads(manifest["path"]["feature_table"].read_text(encoding="utf-8"))
    required = feature_table["input_features" if has_encoder else "model_features"]
    missing = [feature for feature in required if feature not in nodes.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {', '.join(missing)}")
    try:
        values = nodes[required].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Input feature columns must contain numeric values") from error
    if not np.isfinite(values).all():
        raise ValueError("Input feature columns contain missing values or infinity")
    edges = pd.read_csv(edge_path) if edge_path is not None else None
    if has_encoder:
        edge_indices(nodes, edges)
    bundle = load_artifacts(config_path, index, device=device)
    x = prepare_model_input(bundle, nodes, edges)
    model = bundle["model"]
    predicted = model.predict(x) if len(x) else np.array([], dtype=int)
    probabilities = model.predict_proba(x) if len(x) else np.empty((0, len(model.classes_)))
    classes = list(model.classes_)
    result = nodes[[column for column in ("txId", "time_step") if column in nodes]].copy()
    result.insert(0, "input_row", np.arange(len(nodes)))
    result["y_pred"] = predicted
    for label, column in [(0, "probability_normal"), (1, "probability_anomaly")]:
        result[column] = probabilities[:, classes.index(label)] if label in classes else np.zeros(len(nodes))
    output = Path(output_dir) if output_dir is not None else ROOT / "predict_result"
    output.mkdir(parents=True, exist_ok=True)
    output_file = output / f"{index}_{input_path.stem}_predictions.csv"
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Model: {index} ({model_type}); rows: {len(nodes)}")
    print(f"Predictions: {output_file.resolve()}")
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json",
                        help="Model index JSON (default: configs/model.json)")
    parser.add_argument("--model", required=True, help=f"Model type: {', '.join(MODEL_REGISTRY)}")
    parser.add_argument("--index", required=True, help="Entry key in model.json, e.g. seed_42_RF_basic")
    parser.add_argument("--input", required=True, type=Path, help="Node feature CSV")
    parser.add_argument("--edge", type=Path, help="Edge CSV; required for GCN_RF")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="GCN inference device")
    args = parser.parse_args(argv)
    try:
        return predict(args.config, args.model, args.index, args.input, args.edge, device=args.device)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
