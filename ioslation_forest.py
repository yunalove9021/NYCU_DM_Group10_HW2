import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
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
    known = data["y"].notna()

    y_true = data.loc[known, "y"].astype(int).to_numpy()
    y_score_labeled = y_score[known.to_numpy()]

    auc_normal = roc_auc_score(y_true, y_score_labeled)
    auc_reversed = roc_auc_score(y_true, -y_score_labeled)

    print(f"AUROC (original): {auc_normal:.4f}")
    print(f"AUROC (reversed): {auc_reversed:.4f}")
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

class IsolationForestModel:
    def __init__(self, n_estimators=300, random_seed=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_seed,
            n_jobs=-1
        )
        self.threshold = None
        self.best_f1 = None

    def fit(self, X_train):
        """
        使用 Training 的全部資料進行無監督訓練。
        不使用 y label。
        """
        self.model.fit(X_train)

        return self

    def predict_score(self, X):
        """
        Isolation Forest 的 decision_function 越低，
        越可能是 Anomaly，因此取負號。
        讓分數越高代表越可能是 Anomaly。
        但實驗後發現要相反
        """
        return self.model.decision_function(X)

    def select_threshold(self, X_val, y_val):
        """
        使用 Validation 選擇最佳 F1 threshold。
        """
        scores = self.predict_score(X_val)
        y_val = np.asarray(y_val).astype(int)

        thresholds = np.unique(scores)

        best_f1 = -1
        best_threshold = None

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            current_f1 = f1_score(
                y_val, y_pred, zero_division=0
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        self.best_f1 = float(best_f1)

        print(f"Best threshold: {self.threshold:.6f}")
        print(f"Validation best F1: {self.best_f1:.4f}")
        print("\nScore Statistics:")
        print(f"Min:    {scores.min():.6f}")
        print(f"Max:    {scores.max():.6f}")
        print(f"Mean:   {scores.mean():.6f}")
        print(f"Median: {np.median(scores):.6f}")

        print("\nThreshold Statistics:")
        print(f"Selected threshold: {self.threshold:.6f}")
        print(f"Predicted anomaly ratio: {(scores >= self.threshold).mean():.4f}")
        print(f"Actual anomaly ratio: {np.mean(y_val):.4f}")

        return self

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
    ##########################################################################################################################
    features_name = [
        col for col in train.columns
        if col.startswith("feature_")
    ]

    print(f"Number of features: {len(features_name)}")

    normal_mask = (train["y"] == 0)
    X_train = train.loc[normal_mask, features_name]
    print(f"Training on Pure Normal nodes: {len(X_train)} / {len(train)}")

    model = IsolationForestModel(
        n_estimators=n_estimators,
        random_seed=random_seed
    )

    model.fit(X_train)

    # 使用 Validation labeled data 選擇 threshold
    val_labeled = val["y"].notna()

    model.select_threshold(
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

    print(f"Training rows: {len(train)}")
    print(f"Features: {len(features_name)}")
    print(f"Best threshold: {model.threshold:.6f}")
    print(f"Validation best F1: {model.best_f1:.4f}")
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
    output_dir = f'./output/IF_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, n_estimators, random_seed)
    # merge_features(data_dir, output_dir)
