"""Reusable binary-classification prediction, metrics, and ROC exports."""

from pathlib import Path
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)


def evaluate_and_save(model, data, features, name, eval_dir, predict_dir,
                      prefix, label_column="y", run_log=None):
    """Predict all rows; calculate metrics only for known binary labels."""
    eval_dir, predict_dir = Path(eval_dir), Path(predict_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)
    actual = data.get(label_column, pd.Series(float("nan"), index=data.index))
    if not actual.dropna().isin([0, 1]).all():
        raise ValueError(f"{name}: labels must be 0, 1, or missing")
    x = data[features]
    timing = run_log.stage("rf_predict", dataset=name, n_rows=len(data), n_features=len(features),
                           device="cpu") if run_log is not None else nullcontext()
    with timing:
        predicted = model.predict(x) if len(data) else np.array([], dtype=int)
        probabilities = model.predict_proba(x) if len(data) else np.empty((0, len(model.classes_)))
    classes = list(model.classes_)
    anomaly = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(data))
    normal = probabilities[:, classes.index(0)] if 0 in classes else np.zeros(len(data))
    predictions = data[[col for col in ("txId", "time_step") if col in data]].copy()
    predictions["y_true"] = actual
    predictions["y_pred"] = predicted
    predictions["probability_anomaly"] = anomaly
    predictions["probability_normal"] = normal
    predictions.to_csv(predict_dir / f"{prefix}_{name}_predictions.csv",
                       index=False, encoding="utf-8-sig")
    known = actual.notna()
    metrics = {"n_rows": len(data), "n_labeled": int(known.sum()),
               "auroc": None, "pr_auc": None}
    curve = None
    if known.any():
        truth = actual.loc[known].astype(int)
        prediction, score = predicted[known.to_numpy()], anomaly[known.to_numpy()]
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        metrics.update({
            "precision_anomaly": float(precision_score(truth, prediction, zero_division=0)),
            "recall_anomaly": float(recall_score(truth, prediction, zero_division=0)),
            "f1_anomaly": float(f1_score(truth, prediction, zero_division=0)),
            "mcc": float(matthews_corrcoef(truth, prediction)),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        })
        if truth.nunique() == 2:
            fpr, tpr, thresholds = roc_curve(truth, score)
            metrics["auroc"] = float(roc_auc_score(truth, score))
            metrics["pr_auc"] = float(average_precision_score(truth, score))
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
                eval_dir / f"{prefix}_{name}_roc_points.csv", index=False, encoding="utf-8-sig")
            curve = (fpr, tpr)
        else:
            (eval_dir / f"{prefix}_{name}_roc_points.csv").unlink(missing_ok=True)
    else:
        (eval_dir / f"{prefix}_{name}_roc_points.csv").unlink(missing_ok=True)
    return metrics, curve


def save_roc_plot(curves, metrics, output_file):
    """Plot available dataset curves together, without global figure state."""
    output_file = Path(output_file)
    if not curves:
        output_file.unlink(missing_ok=True)
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        for name, (fpr, tpr) in curves.items():
            ax.plot(fpr, tpr, label=f"{name} (AUROC={metrics[name]['auroc']:.4f})")
        ax.plot([0, 1], [0, 1], "--", label="Random (AUROC=0.5)")
        ax.set(xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)",
               title="ROC Curve: Validation / Test")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_file, dpi=180)
    finally:
        plt.close(fig)
