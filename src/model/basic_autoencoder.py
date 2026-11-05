import argparse
import json
import hashlib
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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

def load_or_create_feature_selection(
    train,
    output_dir,
    threshold=0.95,
    force_recompute=False
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 檔名依照 threshold 區分
    threshold_name = str(threshold).replace(".", "")
    json_path = output_dir / f"feature_selection_cor_{threshold_name}.json"

    # 直接讀取已存在的 JSON
    if json_path.exists() and not force_recompute:
        with open(json_path, "r", encoding="utf-8") as f:
            features_name = json.load(f)

        print(f"Loaded feature selection: {json_path}")
        print(f"Number of features: {len(features_name)}")

        return features_name

    # Original features
    original_features = [
        col for col in train.columns
        if col.startswith("feature_")
    ]

    # Training Normal only
    normal_train = train.loc[train["y"] == 0]

    # Correlation filtering
    corr_matrix = normal_train[original_features].corr().abs()

    upper_tri = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > threshold)
    ]

    features_name = [
        f for f in original_features
        if f not in to_drop
    ]

    # Save only feature names
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features_name, f, indent=4, ensure_ascii=False)

    print(f"Saved feature selection: {json_path}")
    print(f"Original features: {len(original_features)}")
    print(f"Dropped features: {len(to_drop)}")
    print(f"Selected features: {len(features_name)}")

    return features_name

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate_and_save(model, data, features_name, name, output):
    """
    同時計算 Original MSE 與 Reversed -MSE 的評估結果。
    僅使用有標籤的資料計算評估指標。
    """

    if model.threshold is None or model.reverse_threshold is None:
        raise ValueError("Both thresholds must be selected first.")

    X = data[features_name]
    mse_scores = model.predict_score(X)

    # Original: MSE
    original_score = mse_scores

    # Reversed: -MSE
    reversed_score = -mse_scores

    # 使用各自 Validation 選出的 threshold
    original_pred = (
        original_score >= model.threshold
    ).astype(int)

    reversed_pred = (
        reversed_score >= model.reverse_threshold
    ).astype(int)

    actual = data["y"]

    known = actual.notna()

    # 輸出 Original Predictions
    pd.DataFrame({
        "time_step": data["time_step"],
        "y_true": actual,
        "y_pred": original_pred,
        "anomaly_score": original_score,
    }).to_csv(
        output / f"{name}_original_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 輸出 Reversed Predictions
    pd.DataFrame({
        "time_step": data["time_step"],
        "y_true": actual,
        "y_pred": reversed_pred,
        "anomaly_score": reversed_score,
    }).to_csv(
        output / f"{name}_reversed_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "n_rows": int(len(data)),
        "n_labeled": int(known.sum()),
        "original": {},
        "reversed": {},
    }

    curve_data = None

    if known.any():

        y_true = actual.loc[known].astype(int).to_numpy()

        original_score_labeled = original_score[known.to_numpy()]
        reversed_score_labeled = reversed_score[known.to_numpy()]

        original_pred_labeled = original_pred[known.to_numpy()]
        reversed_pred_labeled = reversed_pred[known.to_numpy()]

        def calculate_metrics(y_true, y_pred, y_score):

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            result = {
                "precision_anomaly": float(
                    precision_score(
                        y_true,
                        y_pred,
                        pos_label=1,
                        zero_division=0
                    )
                ),
                "recall_anomaly": float(
                    recall_score(
                        y_true,
                        y_pred,
                        pos_label=1,
                        zero_division=0
                    )
                ),
                "f1_anomaly": float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0
                    )
                ),
                "mcc": (
                    float(matthews_corrcoef(y_true, y_pred))
                    if len(np.unique(y_true)) == 2
                    else 0.0
                ),
                "confusion_matrix": {
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                },
            }

            if len(np.unique(y_true)) == 2:

                result["auroc"] = float(
                    roc_auc_score(y_true, y_score)
                )

                result["pr_auc"] = float(
                    average_precision_score(y_true, y_score)
                )

            else:
                result["auroc"] = None
                result["pr_auc"] = None

            return result

        # 計算 Original Metrics
        metrics["original"] = calculate_metrics(
            y_true,
            original_pred_labeled,
            original_score_labeled
        )

        # 計算 Reversed Metrics
        metrics["reversed"] = calculate_metrics(
            y_true,
            reversed_pred_labeled,
            reversed_score_labeled
        )

        if len(np.unique(y_true)) == 2:

            # Original ROC
            fpr_original, tpr_original, thresholds_original = roc_curve(
                y_true,
                original_score_labeled,
                pos_label=1
            )

            # Reversed ROC
            fpr_reversed, tpr_reversed, thresholds_reversed = roc_curve(
                y_true,
                reversed_score_labeled,
                pos_label=1
            )

            pd.DataFrame({
                "fpr": fpr_original,
                "tpr": tpr_original,
                "threshold": thresholds_original,
            }).to_csv(
                output / f"{name}_original_roc_points.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame({
                "fpr": fpr_reversed,
                "tpr": tpr_reversed,
                "threshold": thresholds_reversed,
            }).to_csv(
                output / f"{name}_reversed_roc_points.csv",
                index=False,
                encoding="utf-8-sig",
            )

            curve_data = {
                "original": (fpr_original, tpr_original),
                "reversed": (fpr_reversed, tpr_reversed),
            }

            print(f"\n{name.upper()}")

            print(
                f"Original MSE: "
                f"AUROC={metrics['original']['auroc']:.4f}, "
                f"AP={metrics['original']['pr_auc']:.4f}, "
                f"F1={metrics['original']['f1_anomaly']:.4f}"
            )

            print(
                f"Reversed -MSE: "
                f"AUROC={metrics['reversed']['auroc']:.4f}, "
                f"AP={metrics['reversed']['pr_auc']:.4f}, "
                f"F1={metrics['reversed']['f1_anomaly']:.4f}"
            )

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

class PytorchAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(PytorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoencoderModel:
    def __init__(self, input_dim, latent_dim=32, epochs=50, batch_size=256, lr=1e-3, random_seed=42):
        set_seed(random_seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PytorchAutoencoder(input_dim, latent_dim).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.threshold = None
        self.reverse_threshold = None

        self.best_f1 = None
        self.reverse_best_f1 = None

    def fit(self, X_train):
        """
        使用 Training 正常資料進行無監督特徵標準化與訓練。
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        tensor_x = torch.FloatTensor(X_train_scaled)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Loss: {avg_loss:.6f}"
                )
        return self

    def predict_score(self, X):
        """
        計算 MSE 作為重建誤差。
        針對 Elliptic 分數倒置現象，回傳 -MSE 作為最終的異常分數。
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            # mse_per_sample = torch.mean((reconstructed - tensor_x) ** 2, dim=1)
            l1_per_sample = torch.mean(torch.abs(reconstructed - tensor_x), dim=1)
            
        # scores = mse_per_sample.cpu().numpy()
        scores = l1_per_sample.cpu().numpy()
        return scores

    def select_threshold(self, X_val, y_val):
        """
        使用 Validation 選擇最佳 F1 threshold。
        """
        mse_scores = self.predict_score(X_val)
        y_val = np.asarray(y_val).astype(int)

        if len(y_val) == 0:
            raise ValueError("Validation has no labeled samples.")

        if len(np.unique(y_val)) != 2:
            raise ValueError(
                "Validation must contain both normal and illicit samples."
            )

        def find_best_threshold(scores):

            thresholds = np.unique(scores)

            best_f1 = -1
            best_threshold = None

            for threshold in thresholds:

                y_pred = (scores >= threshold).astype(int)

                current_f1 = f1_score(
                    y_val,
                    y_pred,
                    zero_division=0
                )

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold

            return float(best_threshold), float(best_f1)

        # Original MSE
        self.threshold, self.best_f1 = find_best_threshold(
            mse_scores
        )

        # Reversed -MSE
        self.reverse_threshold, self.reverse_best_f1 = find_best_threshold(
            -mse_scores
        )

        print("\nValidation Threshold Selection")

        print(
            f"Original MSE threshold: {self.threshold:.6f}, "
            f"F1: {self.best_f1:.4f}"
        )

        print(
            f"Reversed -MSE threshold: {self.reverse_threshold:.6f}, "
            f"F1: {self.reverse_best_f1:.4f}"
        )

        print("\nOriginal MSE Statistics:")
        print(f"Min:    {mse_scores.min():.6f}")
        print(f"Max:    {mse_scores.max():.6f}")
        print(f"Mean:   {mse_scores.mean():.6f}")
        print(f"Median: {np.median(mse_scores):.6f}")

        print("\nReversed -MSE Statistics:")
        print(f"Min:    {(-mse_scores).min():.6f}")
        print(f"Max:    {(-mse_scores).max():.6f}")
        print(f"Mean:   {(-mse_scores).mean():.6f}")
        print(f"Median: {np.median(-mse_scores):.6f}")

        return self

def main(data_dir, output_dir, random_seed):
    train = pd.read_csv(data_dir+'nodes_train.csv')
    val = pd.read_csv(data_dir+'nodes_validation.csv')
    test_before_event = pd.read_csv(data_dir+'nodes_test_pre_event.csv')
    test_after_event = pd.read_csv(data_dir+'nodes_test_post_event.csv')
    data_sets = [(train, 'train'), (val, 'val'), (test_before_event, 'test_before_event'), (test_after_event, 'test_after_event')]
    summarize_labels(data_sets, output_dir)
    ##########################################################################################################################
    # features_name = [
    #     col for col in train.columns
    #     if col.startswith("feature_")
    # ]
    features_name = load_or_create_feature_selection(
        train=train,
        output_dir="configs/feature_sets",
        threshold=0.9,
        force_recompute=False
    )

    # 嚴格防漏：預先過濾出純正常節點 (y == 0)
    normal_mask = (train["y"] == 0)
    normal_train = train.loc[normal_mask]
    
    X_train = normal_train[features_name]
    # X_train = train.loc[normal_mask, features_name]
    print(f"Training on Pure Normal nodes: {len(X_train)} / {len(train)}")
    input_dim = len(features_name)

    model = AutoencoderModel(
        input_dim=input_dim,
        latent_dim=64,
        epochs=30,
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
            fpr, tpr = curve_data["original"]
            plt.plot(
                fpr, tpr,
                label=(
                    f"{name.upper()} Original "
                    f"(AUROC={metrics[name]['original']['auroc']:.4f})"
                ),
            )
            fpr, tpr = curve_data["reversed"]

            plt.plot(
                fpr, tpr,
                linestyle="--",
                label=(
                    f"{name.upper()} Reversed "
                    f"(AUROC={metrics[name]['reversed']['auroc']:.4f})"
                )
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

    print(f"Training rows: {len(train)}")
    print(f"Features: {len(features_name)}")
    for name, result in metrics.items():
        original_auc = result["original"].get("auroc")
        reversed_auc = result["reversed"].get("auroc")

        print(
            f"{name.upper()}: "
            f"Original AUROC={original_auc}, "
            f"Reversed AUROC={reversed_auc}"
        )
    print(f"Output: {output.resolve()}")
    return None

if __name__ == '__main__':
    random_seed = 42
    data_dir = f'./Data/'
    output_dir = f'./output/AE_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, random_seed)
