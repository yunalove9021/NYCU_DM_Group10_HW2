"""Average seed probabilities, select a validation threshold, and evaluate val/test."""

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
    # Match existing eval.py convention: leave ranking metrics undefined for single-class labels.
    table["pr_auc"] = float(average_precision_score(truth, scores)) if len(np.unique(truth)) == 2 else np.nan
    table["auroc"] = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else np.nan
    for name, values in [("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp)]:
        table[name] = values.astype(int)
    return table


def choose_threshold(table, metric="f1_anomaly"):
    valid = table.dropna(subset=[metric]).sort_values("threshold")
    if valid.empty:
        raise ValueError("No validation F1 values available for threshold selection")
    # Stable sorting makes exact ties choose the lowest threshold.
    return float(valid.sort_values(metric, ascending=False, kind="stable").iloc[0].threshold)


def average_seed_predictions(paths):
    """Align nodes by txId, validate shared labels, then average probabilities."""
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
            if "time_step" in reference:
                if "time_step" not in frame or not frame.time_step.reset_index(drop=True).equals(reference.time_step):
                    raise ValueError(f"time_step differs across seeds: seed {seed}, {path}")
        probabilities.append(frame.probability_anomaly.to_numpy(dtype=float))
    if reference is None:
        raise ValueError("No seed prediction files supplied")
    columns = [column for column in ("txId", "time_step", "y_true") if column in reference]
    ensemble = reference[columns].copy()
    ensemble["probability_anomaly"] = np.mean(np.stack(probabilities), axis=0)
    ensemble["probability_normal"] = 1 - ensemble.probability_anomaly
    ensemble["seed_count"] = len(paths)
    return ensemble


def plot_curves(curve, best, model, set_name, seed_count, destination):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ["f1_anomaly", "pr_auc"],
                               ["Validation anomaly F1", "Validation PR-AUC (Average Precision)"]):
        ax.plot(curve.threshold, curve[metric], color="#24476A", linewidth=2.3,
                label=f"Mean-probability ensemble ({seed_count} seeds)")
        ax.axvline(best, linestyle="--", color="#D85745", label=f"Best t={best:.2f}")
        ax.set(xlabel="Threshold (positive if mean probability >= threshold)",
               ylabel=metric, title=title, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].text(0.5, 0.08, "PR-AUC uses ensemble probabilities.\nIt does not change with the decision threshold.",
                 transform=axes[1].transAxes, ha="center", fontsize=9)
    fig.suptitle(f"{model} / {set_name} | {seed_count}-seed probability averaging | validation-only selection")
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
    sources = {}

    def averaged(dataset):
        paths = {seed: output / f"{model}_output" / set_name / "predict" /
                 f"{set_name}_seed_{seed}_{model}_{dataset}_predictions.csv" for seed in seeds}
        sources[dataset] = {str(seed): str(path.resolve()) for seed, path in paths.items()}
        return average_seed_predictions(paths)

    # Select once on averaged validation scores before reading test labels.
    validation = averaged("val")
    if set(validation.y_true.dropna().unique()) != {0, 1}:
        raise ValueError(f"Validation requires both labeled classes: {model}/{set_name}")
    curve = evaluate_thresholds(validation, thresholds)
    best = choose_threshold(curve)
    results, averaged_frames = [], {}
    for dataset in datasets:
        frame = validation if dataset == "val" else averaged(dataset)
        values = evaluate_thresholds(frame, [best]).iloc[0].to_dict()
        results.append({"model": model, "feature_set": set_name, "seed_count": len(seeds),
                        "dataset": dataset, **values})
        if save_predictions:
            predictions = frame.copy()
            predictions["threshold"] = best
            predictions["y_pred"] = (predictions.probability_anomaly >= best).astype(int)
            averaged_frames[dataset] = predictions
    result = pd.DataFrame(results).sort_values("dataset")
    metadata = {
        "model": model, "feature_set": set_name, "seeds": seeds, "seed_count": len(seeds),
        "aggregation": "Arithmetic mean of seed probabilities for each txId, within one model and feature set",
        "thresholds": thresholds.tolist(), "best_threshold": best,
        "best_validation_f1": float(curve.loc[curve.threshold == best, "f1_anomaly"].iloc[0]),
        "source_files": sources,
        "rules": {
            "selection": "Maximize anomaly F1 on averaged validation probabilities only; never use test labels for selection.",
            "alignment": "Align all seeds by unique txId; require identical node sets, labels and time_step.",
            "aggregation": "Average probabilities BEFORE computing F1, AP and other metrics; no individual-seed metric averaging.",
            "tie_break": "Choose the lowest threshold among exactly equal maxima.",
            "positive": "mean(probability_anomaly) >= threshold",
            "pr_auc": "Average Precision on averaged probabilities; threshold invariant. Undefined for single-class/no labels.",
            "missing_labels": "Excluded from metrics. No labeled rows -> blank metrics, not zeros.",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    curve.to_csv(destination / "validation_threshold_scan.csv", index=False, encoding="utf-8-sig")
    result.to_csv(destination / "best_threshold_results.csv", index=False, encoding="utf-8-sig")
    for dataset, predictions in averaged_frames.items():
        predictions.to_csv(destination / f"{set_name}_{model}_{len(seeds)}seed_ensemble_{dataset}_predictions.csv",
                           index=False, encoding="utf-8-sig")
    ensemble_info = pd.DataFrame([{
        "model": model, "feature_set": set_name, "seed_count": len(seeds),
        "seeds": ", ".join(map(str, seeds)), "best_threshold": best,
        "best_validation_f1": metadata["best_validation_f1"], "aggregation": metadata["aggregation"],
    }])
    with pd.ExcelWriter(destination / "threshold_results.xlsx", engine="openpyxl") as writer:
        for name, frame in [("Val_scan", curve), ("Best_results", result), ("Ensemble_info", ensemble_info)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(40, max(14, len(str(column[0].value)) + 2))
    plot_curves(curve, best, model, set_name, len(seeds), destination)
    (destination / "threshold_selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Remove only obsolete outputs from the former per-seed metric analysis.
    for name in ["validation_threshold_mean.csv", "best_threshold_per_seed.csv", "best_threshold_results_mean.csv"]:
        (destination / name).unlink(missing_ok=True)
    for policy in ["per_seed_val_best", "shared_mean_val_best"]:
        for path in destination.glob(f"{set_name}_seed_*_{model}_*_{policy}_predictions.csv"):
            path.unlink()
    print(f"{model}/{set_name}: {len(seeds)}-seed probability ensemble, best={best:.2f}, "
          f"val F1={metadata['best_validation_f1']:.4f}; {destination}", flush=True)
    return result


def model_type(value):
    value = value.replace("-", "_")
    if value not in {"RF", "GCN_RF"}:
        raise argparse.ArgumentTypeError(f"Model '{value}' is not supported; use RF or GCN_RF")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output-dir", type=Path, help="Input training output root; defaults to YAML output")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "Analysis" / "threshold")
    parser.add_argument("--models", nargs="+", type=model_type, default=["RF", "GCN_RF"])
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seed IDs; defaults to YAML random_seeds")
    parser.add_argument("--expected-seeds", type=int, default=5, help="Required ensemble size (default: 5)")
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--save-predictions", action="store_true", help="Also export mean-probability ensemble predictions")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output = args.output_dir.resolve() if args.output_dir else (config_path.parent / config["output"]).resolve()
        seeds = args.seeds if args.seeds is not None else config["random_seeds"]
        seeds = [seeds] if type(seeds) is int else seeds
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Require a nonempty list of unique nonnegative seeds")
        seeds = sorted(seeds)
        if args.expected_seeds < 2 or len(seeds) != args.expected_seeds:
            raise ValueError(f"Require {args.expected_seeds} distinct ensemble seeds (minimum 2), got {seeds}")
        models = list(dict.fromkeys(args.models))
        sets = args.feature_sets or sorted({path.name for model in models for path in (output / f"{model}_output").glob("*") if path.is_dir()})
        if not sets:
            raise ValueError(f"No RF/GCN_RF output feature sets found: {output}")
        tests = config["data"]["test_files"]
        tests = [tests] if isinstance(tests, str) else tests
        names = list(tests) if isinstance(tests, dict) else [Path(name).stem for name in tests]
        datasets = ["val", *names]
        if len(datasets) != len(set(datasets)):
            raise ValueError("Validation and test dataset names must be unique")
        for name in [*sets, *datasets]:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid feature set/dataset name: {name}")
        thresholds = threshold_grid(args.start, args.end, args.step)
        # Fail explicitly when an expected seed or test export is missing.
        for model in models:
            for set_name in sets:
                for seed in seeds:
                    for dataset in datasets:
                        path = output / f"{model}_output" / set_name / "predict" / f"{set_name}_seed_{seed}_{model}_{dataset}_predictions.csv"
                        if not path.is_file():
                            raise FileNotFoundError(path)
        for model in models:
            for set_name in sets:
                analyze_group(output, args.save_dir.resolve() / model / set_name, model, set_name,
                              seeds, datasets, thresholds, args.save_predictions)
        return args.save_dir.resolve()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
