"""Summarize configured training results and seed averages into an Excel report."""

import argparse
import json
from numbers import Real
from pathlib import Path
import re

import pandas as pd
import yaml
from openpyxl.styles import Alignment, Font, PatternFill

from src.config import resolve_feature_sets


ROOT = Path(__file__).resolve().parent
METRICS = ["n_rows", "n_labeled", "auroc", "pr_auc", "precision_anomaly",
           "recall_anomaly", "f1_anomaly", "mcc", "confusion_matrix_TN",
           "confusion_matrix_FP", "confusion_matrix_FN", "confusion_matrix_TP"]
DESCRIPTIONS = {
    "params.n_estimators": "Random Forest 決策樹數量",
    "params.class_weight": "RF 類別權重；balanced 依訓練類別數量調整",
    "params.n_jobs": "RF 平行工作數；-1 使用可用 CPU 核心",
    "encoder.name": "圖特徵編碼器類型",
    "encoder.device": "GCN 訓練裝置；auto 自動選擇 CPU/GPU",
    "encoder.undirected": "是否加入反向邊；false 保留原始方向",
    "encoder.params.hidden_dim": "第一層 GCN 輸出維度",
    "encoder.params.embedding_dim": "Node embedding 維度",
    "encoder.params.num_layers": "GCN 卷積層數，不含分類頭",
    "encoder.params.dropout": "第一層之後的 dropout 比例",
    "encoder.params.learning_rate": "GCN Adam optimizer 學習率",
    "encoder.params.epochs": "GCN 最大訓練 epochs",
    "encoder.params.patience": "Validation loss 未改善時的 early stopping 等待 epochs",
    "encoder.loss.name": "GCN 監督式分類 loss",
    "encoder.loss.class_weights": "GCN balanced 權重：N_labeled / (2 * N_class)",
}


def flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def build_report(config_path, output_path=None):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = (config_path.parent / config["output"]).resolve()
    seeds = config["random_seeds"]
    if type(seeds) is int:
        seeds = [seeds]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds must contain unique integers")
    settings = dict(config["data"])
    resolve_feature_sets(settings, config_path.parent)
    tests = settings["test_files"]
    if isinstance(tests, str):
        tests = [tests]
    test_names = list(tests) if isinstance(tests, dict) else [Path(name).stem for name in tests]
    datasets = ["val", *test_names]
    if len(datasets) != len(set(datasets)):
        raise ValueError("Dataset names must be unique")
    rows, statuses, parameters = [], [], []
    for model_name, model_settings in sorted(config["models"].items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
            raise ValueError(f"Invalid model name: {model_name}")
        for name, value in flatten(model_settings).items():
            parameters.append({"model": model_name, "parameter": name,
                               "value": json.dumps(value, ensure_ascii=False),
                               "description": DESCRIPTIONS.get(name, "模型設定參數，依 YAML 原值列出"),
                               "source": str(config_path)})
        parameters.append({"model": model_name, "parameter": "random_seeds",
                           "value": json.dumps(seeds), "description": "各次訓練的 random_state；GCN 同時設定 NumPy/PyTorch seed",
                           "source": str(config_path)})
        for feature_set in settings["feature_sets"]:
            set_name = feature_set["set_name"]
            for seed in sorted(seeds):
                prefix = f"{set_name}_seed_{seed}_{model_name}"
                metrics_path = output_root / f"{model_name}_output" / set_name / "eval" / f"{prefix}_metrics.json"
                base = {"model": model_name, "feature_set": set_name, "seed": seed}
                error, metrics = None, {}
                if not metrics_path.is_file():
                    error = ("missing_file", "找不到 metrics 檔案")
                else:
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if not isinstance(metrics, dict):
                            raise ValueError("Metrics root must be an object")
                    except (ValueError, OSError) as exc:
                        error = ("invalid_file", str(exc))
                for dataset in datasets:
                    status = {**base, "dataset": dataset, "metrics_file": str(metrics_path)}
                    if error:
                        statuses.append({**status, "status": error[0], "description": error[1]})
                        continue
                    if dataset not in metrics:
                        statuses.append({**status, "status": "missing_dataset", "description": "Metrics 中缺少此資料集"})
                        continue
                    if not isinstance(metrics[dataset], dict):
                        statuses.append({**status, "status": "invalid_metrics", "description": "資料集 metrics 必須為物件"})
                        continue
                    values = flatten(metrics[dataset])
                    invalid = [key for key, value in values.items()
                               if value is not None and (not isinstance(value, Real) or isinstance(value, bool))]
                    if invalid:
                        statuses.append({**status, "status": "invalid_metrics", "description": f"非數值 metrics: {invalid}"})
                        continue
                    row = {**base, "dataset": dataset,
                           **{key.replace(".", "_"): value for key, value in values.items()},
                           "metrics_file": str(metrics_path)}
                    rows.append(row)
                    statuses.append({**status, "status": "ok", "description": "已讀取"})
    detail = pd.DataFrame(rows)
    keys = ["model", "feature_set", "dataset"]
    extra = sorted({key for row in rows for key in row} - set(keys + ["seed", "metrics_file"] + METRICS))
    metric_columns = METRICS + extra
    detail = detail.reindex(columns=keys + ["seed"] + metric_columns + ["metrics_file"])
    for metric in metric_columns:
        detail[metric] = pd.to_numeric(detail[metric], errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    detail = detail.sort_values(keys + ["seed"]).reset_index(drop=True)
    summary_rows = []
    # Include groups with no available results so missing runs remain visible.
    for model_name in sorted(config["models"]):
        for feature_set in settings["feature_sets"]:
            for dataset in datasets:
                group = detail[(detail.model == model_name) & (detail.feature_set == feature_set["set_name"]) & (detail.dataset == dataset)]
                summary = {"model": model_name, "feature_set": feature_set["set_name"], "dataset": dataset,
                           "expected_seeds": len(seeds), "available_seeds": len(group),
                           "seeds": ", ".join(str(seed) for seed in group.seed)}
                for metric in metric_columns:
                    values = group[metric].dropna()
                    summary[f"{metric}_mean"] = values.mean()
                    summary[f"{metric}_std"] = values.std(ddof=1)
                summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows).sort_values(keys)
    checks = pd.DataFrame(statuses).sort_values(keys + ["seed"])
    hyperparameters = pd.DataFrame(parameters).sort_values(["model", "parameter"])
    notes = pd.DataFrame([
        ("config", str(config_path)), ("output", str(output_root)),
        ("排序", "所有結果依 model、feature_set、dataset、seed 排序"),
        ("平均", "同模型、特徵集、資料集的 seed 等權算術平均；不跨資料集或特徵集合併"),
        ("缺失", "缺失或無法計算的指標不視為 0；各指標僅以非缺失值計算，可回查 Metrics_by_seed"),
        ("Seed 數", "expected_seeds 為設定 seed 數，available_seeds 為已讀取結果數；不代表每個指標皆有值"),
        ("標準差", "_std 為樣本標準差（ddof=1）；有效 seed 少於 2 時留白"),
        ("Hyperparameters", "來自目前 YAML，不是歷史模型訓練設定快照；設定改動後可能與既有結果不同"),
        ("Metrics", "AUROC、PR-AUC、precision、recall、F1、MCC、TN/FP/FN/TP 與樣本數"),
        ("混淆矩陣", "TN/FP/FN/TP 的平均為各 seed 計數平均，不是合併後重新計算"),
    ], columns=["item", "description"])
    target = Path(output_path) if output_path is not None else ROOT / "Analysis" / "training_results.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in [("Metrics_by_seed", detail), ("Seed_summary", summary),
                            ("Hyperparameters", hyperparameters), ("Run_status", checks), ("Notes", notes)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(color="FFFFFF", bold=True)
                cell.fill = PatternFill("solid", fgColor="24476A")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.row_dimensions[1].height = 32
            for column in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(60, max(14, width + 2))
                for cell in column[1:]:
                    cell.alignment = Alignment(vertical="top")
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"
    print(f"Read {len(detail)} dataset/seed results; {int((checks.status != 'ok').sum())} missing/invalid entries")
    print(f"Excel: {target.resolve()}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "Analysis" / "training_results.xlsx")
    args = parser.parse_args(argv)
    try:
        return build_report(args.config, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
