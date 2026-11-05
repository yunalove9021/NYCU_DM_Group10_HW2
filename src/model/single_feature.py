import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    matthews_corrcoef
)
from sklearn import set_config
set_config(display="text")
random_seed = 42

def evaluate_and_save(model, data, features_name, name, output):
    """輸出所有資料列的預測，僅用有標籤的資料計算評估指標。"""
    x = data[features_name]
    ##########################################################################################################################
    y_score = model.predict_score(x)
    predicted = model.predict(x)
    ##########################################################################################################################

    # classes_ 的順序不可自行假定；0=正常，1=異常
    # classes = list(model.classes_)
    # probability_normal = probabilities[:, classes.index(0)]
    # probability_anomaly = probabilities[:, classes.index(1)]

    actual = (
        data["y"]
        if "y" in data.columns
        else pd.Series(float("nan"), index=data.index)
    )

    pd.DataFrame({
        "time_step": data["time_step"],
        "y_true": actual,
        "y_pred": predicted,
        "anomaly_score": y_score,
    }).to_csv(
        output / f"{name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    known = actual.notna()
    metrics = {
        "n_rows": int(len(data)),
        "n_labeled": int(known.sum()),
        "auroc": None,
    }
    curve_data = None

    if known.any():
        y_true = actual.loc[known].astype(int)
        y_pred = predicted[known.to_numpy()]
        y_score_labeled = y_score[known.to_numpy()]

        # labels=[0, 1] 時，矩陣為 [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        metrics.update({
            "precision_anomaly": float(precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            "recall_anomaly": float(recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            "f1_anomaly": float(f1_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            },
        })

        # AUROC/ROC 必須同時有 0 與 1 兩種真實標籤
        if y_true.nunique() == 2:
            fpr, tpr, thresholds = roc_curve(
                y_true, y_score_labeled, pos_label=1
            )
            metrics["auroc"] = float(roc_auc_score(y_true, y_score_labeled))
            # PR-AUC (Average Precision)
            metrics["pr_auc"] = float(
                average_precision_score(y_true, y_score_labeled)
            )
            pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            }).to_csv(
                output / f"{name}_roc_points.csv",
                index=False,
                encoding="utf-8-sig",
            )
            curve_data = (fpr, tpr)

    return metrics, curve_data

def summarize_labels(data_sets, output_dir):
    """統計單一資料集的正常、異常及未標記樣本數量。"""
    output_dir = Path(output_dir)
    all_summary = []
    for data, name in data_sets:
        y = (
            data["y"]
            if "y" in data.columns
            else pd.Series(float("nan"), index=data.index)
        )
        summary = {
            "dataset": name,
            "total": int(len(data)),
            "labeled": int(y.notna().sum()),
            "normal_0": int((y == 0).sum()),
            "anomaly_1": int((y == 1).sum()),
            "unlabeled_nan": int(y.isna().sum()),
        }
        all_summary.append(summary)
    summary = pd.DataFrame(all_summary)
    summary.to_excel(
        output_dir / f"label_summary.xlsx",
        index=False
    )
    return summary

class SingleFeatureModel:
    def __init__(self, feature_name="feature_5"):
        self.feature_name = feature_name
        self.threshold = None
        self.direction = None
        self.best_f1 = None

    def fit(self, X, y):
        """
        使用 Validation 決定 score direction 和最佳 threshold。
        """
        scores = X[self.feature_name].to_numpy()
        y = np.asarray(y).astype(int)

        # 比較 feature 越大、越小哪個方向更能辨識 Anomaly
        auc_positive = roc_auc_score(y, scores)
        auc_negative = roc_auc_score(y, -scores)

        if auc_positive >= auc_negative:
            self.direction = 1
            anomaly_scores = scores
        else:
            self.direction = -1
            anomaly_scores = -scores

        # 在 Validation 選擇最佳 F1 threshold
        thresholds = np.unique(anomaly_scores)

        best_f1 = -1
        best_threshold = None

        for threshold in thresholds:
            y_pred = (anomaly_scores >= threshold).astype(int)
            current_f1 = f1_score(y, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        self.best_f1 = float(best_f1)

        print(f"Feature: {self.feature_name}")
        print(f"Direction: {'Higher = Anomaly' if self.direction == 1 else 'Lower = Anomaly'}")
        print(f"Best threshold: {self.threshold:.6f}")
        print(f"Validation best F1: {self.best_f1:.4f}")

        return self

    def predict_score(self, X):
        scores = X[self.feature_name].to_numpy()
        return self.direction * scores

    def predict(self, X):
        scores = self.predict_score(X)
        return (scores >= self.threshold).astype(int)

def main(data_dir, output_dir, n_estimators, random_seed):
    train = pd.read_csv(data_dir+'nodes_train.csv')
    val = pd.read_csv(data_dir+'nodes_validation.csv')
    test_before_event = pd.read_csv(data_dir+'nodes_test_pre_event.csv')
    test_after_event = pd.read_csv(data_dir+'nodes_test_post_event.csv')
    data_sets = [(train, 'train'), (val, 'val'), (test_before_event, 'test_before_event'), (test_after_event, 'test_after_event')]
    summarize_labels(data_sets, output_dir)
    features_name = ["feature_18"]
    print(len(train))
    val_labeled = val["y"].notna()
    ##########################################################################################################################
    model = SingleFeatureModel(feature_name="feature_18")
    model.fit(
        val.loc[val_labeled, features_name],
        val.loc[val_labeled, "y"]
    )
    ##########################################################################################################################
    output = Path(output_dir)
    metrics = {}
    plt.figure(figsize=(7, 6))
    has_curve = False
    for name, data in (("val", val), ("test_before_event", test_before_event), ("test_after_event", test_after_event)):
        metrics[name], curve_data = evaluate_and_save(
            model, data, features_name, name, output
        )
        if curve_data is not None:
            fpr, tpr = curve_data
            plt.plot(
                fpr, tpr,
                label=f"{name.upper()} (AUROC={metrics[name]['auroc']:.4f})",
            )
            has_curve = True
    if has_curve:
        plt.plot([0, 1], [0, 1], "--", label="Random (AUROC=0.5)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve: Validation / Test")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "roc_curve.png", dpi=180)
    plt.close()

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # importance = pd.DataFrame({
    #     "feature": features_name,
    #     "importance": model.feature_importances_,
    # }).sort_values("importance", ascending=False).reset_index(drop=True)
    # importance.insert(0, "rank", importance.index + 1)
    # importance.to_csv(
    #     output / "feature_importance.csv", index=False, encoding="utf-8-sig"
    # )

    # joblib.dump(
    #     {"model": model, "features": features_name},
    #     output / "model.joblib",
    # )

    print(f"Labeled validation rows: {int(val_labeled.sum())}")
    # print(f"Features: {len(features_name)}")
    print(f"Selected feature: {model.feature_name}")
    print(f"Direction: {model.direction}")
    print(f"Threshold: {model.threshold:.6f}")
    for name, result in metrics.items():
        auc_text = (
            f"{result['auroc']:.4f}"
            if result["auroc"] is not None
            else "unavailable"
        )
        print(f"{name.upper()}: labeled={result['n_labeled']}, AUROC={auc_text}")
    print(f"Output: {output.resolve()}")
    return None

def merge_features(data_dir, output_dir):
    log_reg = pd.read_csv(fr"{data_dir}Logistic_Regression.csv")
    rf_feature_important = pd.read_csv(fr"{output_dir}feature_importance.csv")
    df = pd.merge(log_reg, rf_feature_important, how= 'left')
    df = df.astype({'rank':'Int64'})
    df.to_excel(fr"{output_dir}feature_merge_analysis.xlsx")
    return None

if __name__ == '__main__':
    random_seed = 42
    n_estimators = 300
    data_dir = f'./Data/'
    output_dir = f'./output/SF_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, n_estimators, random_seed)
    # merge_features(data_dir, output_dir)
