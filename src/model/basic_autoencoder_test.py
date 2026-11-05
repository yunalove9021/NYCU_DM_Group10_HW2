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
        self.random_seed = random_seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.reverse_threshold = None

        self.best_f1 = None
        self.reverse_best_f1 = None
        self.best_ap = None
        self.reverse_best_ap = None

    def fit(self, X_train, X_val):
        """
        使用 Training 正常資料進行無監督特徵標準化與訓練。
        """
        self.model = PytorchAutoencoder(self.input_dim, self.latent_dim).to(self.device)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        tensor_x = torch.FloatTensor(X_train_scaled)
        dataset = TensorDataset(tensor_x)
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        tensor_val = torch.FloatTensor(X_val_scaled).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        # early stop
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
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
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                val_reconstructed = self.model(tensor_val)
                val_loss = criterion(val_reconstructed, tensor_val).item()
            
            # Early Stopping 邏輯
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                import copy
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Loss: {avg_loss:.6f}"
                )
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}!")
                break
        # 還原泛化能力最佳的權重
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
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
        使用 Validation 選擇最佳 AP threshold。
        """
        mae_scores = self.predict_score(X_val)
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
        self.best_ap = average_precision_score(
            y_val,
            mae_scores
        )
        self.threshold, self.best_f1 = find_best_threshold(
            mae_scores
        )

        # Reversed -MSE
        reversed_scores = -mae_scores
        self.reverse_best_ap = average_precision_score(
            y_val,
            reversed_scores
        )
        self.reverse_threshold, self.reverse_best_f1 = find_best_threshold(
            reversed_scores
        )

        print("\nValidation Threshold Selection")

        print("\nOriginal MAE:")
        print(f"AP:        {self.best_ap:.4f}")
        print(f"Threshold: {self.threshold:.6f}")
        print(f"F1:        {self.best_f1:.4f}")

        print("\nReversed -MAE:")
        print(f"AP:        {self.reverse_best_ap:.4f}")
        print(f"Threshold: {self.reverse_threshold:.6f}")
        print(f"F1:        {self.reverse_best_f1:.4f}")

        print("\nOriginal MAE Statistics:")
        print(f"Min:    {mae_scores.min():.6f}")
        print(f"Max:    {mae_scores.max():.6f}")
        print(f"Mean:   {mae_scores.mean():.6f}")
        print(f"Median: {np.median(mae_scores):.6f}")

        print("\nReversed -MAE Statistics:")
        print(f"Min:    {reversed_scores.min():.6f}")
        print(f"Max:    {reversed_scores.max():.6f}")
        print(f"Mean:   {reversed_scores.mean():.6f}")
        print(f"Median: {np.median(reversed_scores):.6f}")

        return self

def main(data_dir, output_dir, random_seed):
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

    normal_mask_train = (train["y"] == 0)
    X_train_normal = train.loc[normal_mask_train, features_name]
    input_dim = len(features_name)
    val_labeled = val["y"].notna()
    normal_mask_val = (val["y"] == 0) & val_labeled
    X_val_normal = val.loc[normal_mask_val, features_name]

    param_grid = {
        'latent_dim': [32, 64],
        'batch_size': [128, 256],
        'epochs':[50, 100],
        'lr': [1e-3, 5e-4]
    }
    
    import itertools
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    
    best_overall_ap = -1
    best_params = None
    best_model = None
    
    print(f"Starting Grid Search over {len(combinations)} combinations...")
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing parameters: {params}")
        
        # 初始化模型 (epochs 設較大以供 Early Stopping 發揮)
        current_model = AutoencoderModel(
            input_dim=input_dim,
            latent_dim=params['latent_dim'],
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            lr=params['lr'],
            random_seed=random_seed
        )
        
        # 訓練模型 (傳入 X_train_normal 與 X_val_normal 進行 Early Stopping)
        current_model.fit(X_train_normal, X_val_normal)
        # 只計算 Validation AP 進行模型選擇
        val_scores = current_model.predict_score(
            val.loc[val_labeled, features_name]
        )

        val_ap = average_precision_score(
            val.loc[val_labeled, "y"].astype(int),
            -val_scores
        )
        print(f"Validation Reversed AP: {val_ap:.4f}")
        
        if val_ap > best_overall_ap:
            best_overall_ap = val_ap
            best_params = params

            import copy
            best_model = copy.deepcopy(current_model)
    
    # 最佳模型確定後，才選擇分類 threshold
    best_model.select_threshold(
        val.loc[val_labeled, features_name],
        val.loc[val_labeled, "y"]
    )

            
    print("\n==================================================")
    print("Grid Search Completed!")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Reversed AP: {best_overall_ap:.4f}")
    print(f"Best Validation Reversed F1: {best_model.reverse_best_f1:.4f}")
    print(f"Best Reversed Threshold: {best_model.reverse_threshold:.6f}")
    ##########################################################################################################################
    output = Path(output_dir)
    metrics = {}
    plt.figure(figsize=(7, 6))
    has_curve = False
    for name, data in (("val", val), ("test_before_event", test_before_event), ("test_after_event", test_after_event)):
        metrics[name], curve_data = evaluate_and_save(
            best_model, data, features_name, name, output
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
    output_dir = f'./output/AE_best_output/'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main(data_dir, output_dir, random_seed)
