"""Average seed probabilities (or handle single model), select a validation threshold, and evaluate val/test."""

import argparse
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parent
METRICS = ["precision_anomaly", "recall_anomaly", "f1_anomaly", "mcc", "pr_auc", "auroc",
           "TN", "FP", "FN", "TP"]


def threshold_grid(start=0.5, end=1.0, step=0.01):
    from decimal import Decimal
    start, end, step = map(lambda value: Decimal(str(value)), (start, end, step))
    if not (0 <= start <= end <= 1) or step <= 0:
        raise ValueError("Require 0 <= start <= end <= 1 and step > 0")
    count = (end - start) / step
    if count != count.to_integral_value():
        raise ValueError("Threshold range must be divisible by step")
    if count > 10000:
        raise ValueError("Too many thresholds; increase step")
    return np.array([float(start + index * step) for index in range(int(count) + 1)])


def read_predictions(path):
    frame = pd.read_csv(path)
    if not {"y_true", "probability_anomaly"} <= set(frame):
        raise ValueError(f"Require y_true and probability_anomaly columns: {path}")
    labels = pd.to_numeric(frame.y_true, errors="raise")
    scores = pd.to_numeric(frame.probability_anomaly, errors="raise")
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError(f"Labels must be 0, 1, or missing: {path}")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError(f"Anomaly probabilities must be finite and in [0, 1]: {path}")
    frame["y_true"], frame["probability_anomaly"] = labels, scores
    return frame


def evaluate_thresholds(frame, thresholds):
    """Vectorized confusion counts; AP/AUROC always use the original scores."""
    known = frame.y_true.notna()
    truth = frame.loc[known, "y_true"].to_numpy(dtype=int)
    scores = frame.loc[known, "probability_anomaly"].to_numpy(dtype=float)
    count = len(thresholds)
    table = pd.DataFrame({"threshold": thresholds, "n_rows": len(frame), "n_labeled": len(truth)})
    if not len(truth):
        for metric in METRICS:
            table[metric] = np.nan
        return table
    predicted = scores[:, None] >= np.asarray(thresholds)[None, :]
    positive = truth[:, None] == 1
    tp = (predicted & positive).sum(axis=0).astype(float)
    fp = (predicted & ~positive).sum(axis=0).astype(float)
    fn, tn = float((truth == 1).sum()) - tp, float((truth == 0).sum()) - fp
    divide = lambda numerator, denominator: np.divide(numerator, denominator, out=np.zeros(count), where=denominator != 0)
    table["precision_anomaly"] = divide(tp, tp + fp)
    table["recall_anomaly"] = divide(tp, tp + fn)
    table["f1_anomaly"] = divide(2 * tp, 2 * tp + fp + fn)
    table["mcc"] = divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    table["pr_auc"] = float(average_precision_score(truth, scores)) if len(np.unique(truth)) == 2 else np.nan
    table["auroc"] = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else np.nan
    for name, values in [("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp)]:
        table[name] = values.astype(int)
    return table


def choose_threshold(table, metric="f1_anomaly"):
    valid = table.dropna(subset=[metric]).sort_values("threshold")
    if valid.empty:
        raise ValueError("No validation F1 values available for threshold selection")
    return float(valid.sort_values(metric, ascending=False, kind="stable").iloc[0].threshold)


def get_model_predictions(output, model, set_name, dataset, seeds):
    """支援多 seed 平均，若為 single_feature 則直接讀取單一 seed 檔案"""
    if "single_feature" in model:
        # single_feature 通常固定為 seed 42 或讀取清單中的第一個 seed
        seed = seeds[0]
        path = output / f"{model}_output" / set_name / "predict" / f"{set_name}_seed_{seed}_{model}_{dataset}_predictions.csv"
        if not path.is_file():
            # 兼容可能的檔名變化
            matches = list((output / f"{model}_output" / set_name / "predict").glob(f"*_{dataset}_predictions.csv"))
            if matches:
                path = matches[0]
            else:
                raise FileNotFoundError(path)
        frame = read_predictions(path)
        sources = {str(seed): str(path.resolve())}
        return frame, sources
    else:
        # 原本的 multi-seed 邏輯
        paths = {seed: output / f"{model}_output" / set_name / "predict" /
                       f"{set_name}_seed_{seed}_{model}_{dataset}_predictions.csv" for seed in seeds}
        sources = {str(seed): str(path.resolve()) for seed, path in paths.items()}
        
        reference, probabilities = None, []
        for seed, path in paths.items():
            frame = read_predictions(path)
            if "txId" not in frame or frame.txId.isna().any() or frame.txId.duplicated().any():
                raise ValueError(f"Seed {seed} requires unique, nonmissing txId: {path}")
            if reference is None:
                reference = frame.reset_index(drop=True)
            else:
                if len(frame) != len(reference) or set(frame.txId) != set(reference.txId):
                    raise ValueError(f"Node IDs differ across seeds: seed {seed}, {path}")
                frame = frame.set_index("txId").loc[reference.txId].reset_index()
                if not np.array_equal(frame.y_true.to_numpy(dtype=float),
                                      reference.y_true.to_numpy(dtype=float), equal_nan=True):
                    raise ValueError(f"Labels differ across seeds after txId alignment: seed {seed}, {path}")
            probabilities.append(frame.probability_anomaly.to_numpy(dtype=float))
        columns = [column for column in ("txId", "time_step", "y_true") if column in reference]
        ensemble = reference[columns].copy()
        ensemble["probability_anomaly"] = np.mean(np.stack(probabilities), axis=0)
        ensemble["probability_normal"] = 1 - ensemble.probability_anomaly
        ensemble["seed_count"] = len(seeds)
        return ensemble, sources


def plot_curves(curve, best, model, set_name, seed_count, destination):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    label_text = f"Single feature baseline" if "single_feature" in model else f"Mean-probability ensemble ({seed_count} seeds)"
    for ax, metric, title in zip(axes, ["f1_anomaly", "pr_auc"],
                                 ["Validation anomaly F1", "Validation PR-AUC (Average Precision)"]):
        ax.plot(curve.threshold, curve[metric], color="#24476A", linewidth=2.3, label=label_text)
        ax.axvline(best, linestyle="--", color="#D85745", label=f"Best t={best:.2f}")
        ax.set(xlabel="Threshold (probability_anomaly >= threshold)",
               ylabel=metric, title=title, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].text(0.5, 0.08, "PR-AUC uses model probabilities.\nIt does not change with the decision threshold.",
                 transform=axes[1].transAxes, ha="center", fontsize=9)
    fig.suptitle(f"{model} / {set_name} | threshold optimization")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    try:
        for extension in ["png", "pdf"]:
            with tempfile.NamedTemporaryFile(dir=destination, suffix=f".{extension}", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                fig.savefig(temporary, format=extension, dpi=220, bbox_inches="tight")
                temporary.replace(destination / f"validation_threshold_curves.{extension}")
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        plt.close(fig)


def analyze_group(output, destination, model, set_name, seeds, datasets, thresholds, save_predictions=False):
    # 取得驗證集預測
    validation, sources_val = get_model_predictions(output, model, set_name, "val", seeds)
    sources = {"val": sources_val}

    if set(validation.y_true.dropna().unique()) != {0, 1}:
        raise ValueError(f"Validation requires both labeled classes: {model}/{set_name}")
    
    curve = evaluate_thresholds(validation, thresholds)
    best = choose_threshold(curve)
    results, averaged_frames = [], {}
    
    for dataset in datasets:
        if dataset == "val":
            frame = validation
        else:
            frame, sources_ds = get_model_predictions(output, model, set_name, dataset, seeds)
            sources[dataset] = sources_ds
            
        values = evaluate_thresholds(frame, [best]).iloc[0].to_dict()
        results.append({"model": model, "feature_set": set_name, "seed_count": len(seeds) if "single_feature" not in model else 1,
                        "dataset": dataset, **values})
        if save_predictions:
            predictions = frame.copy()
            predictions["threshold"] = best
            predictions["y_pred"] = (predictions.probability_anomaly >= best).astype(int)
            averaged_frames[dataset] = predictions
            
    result = pd.DataFrame(results).sort_values("dataset")
    metadata = {
        "model": model, "feature_set": set_name, "seeds": seeds, "seed_count": len(seeds) if "single_feature" not in model else 1,
        "thresholds": thresholds.tolist(), "best_threshold": best,
        "best_validation_f1": float(curve.loc[curve.threshold == best, "f1_anomaly"].iloc[0]),
        "source_files": sources,
        "rules": {
            "selection": "Maximize anomaly F1 on validation probabilities only.",
            "positive": "probability_anomaly >= threshold",
            "pr_auc": "Average Precision on probabilities; threshold invariant.",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    curve.to_csv(destination / "validation_threshold_scan.csv", index=False, encoding="utf-8-sig")
    result.to_csv(destination / "best_threshold_results.csv", index=False, encoding="utf-8-sig")
    
    for dataset, predictions in averaged_frames.items():
        predictions.to_csv(destination / f"{set_name}_{model}_ensemble_{dataset}_predictions.csv",
                           index=False, encoding="utf-8-sig")
                           
    ensemble_info = pd.DataFrame([{
        "model": model, "feature_set": set_name, "seed_count": metadata["seed_count"],
        "seeds": ", ".join(map(str, seeds)), "best_threshold": best,
        "best_validation_f1": metadata["best_validation_f1"],
    }])
    
    with pd.ExcelWriter(destination / "threshold_results.xlsx", engine="openpyxl") as writer:
        for name, frame in [("Val_scan", curve), ("Best_results", result), ("Ensemble_info", ensemble_info)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(40, max(14, len(str(column[0].value)) + 2))
                
    plot_curves(curve, best, model, set_name, metadata["seed_count"], destination)
    (destination / "threshold_selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        
    print(f"{model}/{set_name}: best threshold={best:.4f}, val F1={metadata['best_validation_f1']:.4f}; {destination}", flush=True)
    return result


def model_type(value):
    value = value.replace("-", "_")
    # 支援單一特徵與其他模型
    if not (value in {"RF", "GCN_RF"} or "single_feature" in value):
        raise argparse.ArgumentTypeError(f"Model '{value}' is not supported.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output-dir", type=Path, help="Input training output root; defaults to YAML output")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "Analysis" / "threshold")
    parser.add_argument("--models", nargs="+", type=model_type, default=["single_feature_feature_60"])
    parser.add_argument("--feature-sets", nargs="+", default=["basic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--expected-seeds", type=int, default=1, help="Required ensemble size")
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)
    
    try:
        config_path = args.config.resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output = args.output_dir.resolve() if args.output_dir else (config_path.parent / config["output"]).resolve()
        seeds = args.seeds
        models = list(dict.fromkeys(args.models))
        sets = args.feature_sets
        
        tests = config["data"]["test_files"]
        tests = [tests] if isinstance(tests, str) else tests
        names = list(tests) if isinstance(tests, dict) else [Path(name).stem for name in tests]
        datasets = ["val", *names]
        
        thresholds = threshold_grid(args.start, args.end, args.step)
        
        for model in models:
            for set_name in sets:
                analyze_group(output, args.save_dir.resolve() / model / set_name, model, set_name,
                            seeds, datasets, thresholds, args.save_predictions)
        return args.save_dir.resolve()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()