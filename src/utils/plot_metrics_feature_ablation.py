"""Visualize model performances: Single Feature as 1 bar, and other models grouped by feature sets."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. 定義要呈現的模型與所屬 Tier 顏色
# ==============================================================================
TIER_COLORS = {
    "Tier 1 (Baselines)": "#B0B0B0",      # 灰色
    "Tier 2 (Unsupervised)": "#4C72B0",   # 藍色
    "Tier 3 (Deep/Graph)": "#DD8452"      # 橘色
}

METRICS_TO_PLOT = [
    {"col": "pr_auc", "title": "PR-AUC"},
    {"col": "f1_anomaly", "title": "Anomaly F1-Score"},
    {"col": "mcc", "title": "Matthews Correlation Coefficient (MCC)"}
]

def load_data(excel_path, target_dataset):
    """讀取 Excel 總表的 Seed_summary 分頁並過濾 Dataset"""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 檔案: {excel_path}")
        
    df = pd.read_excel(excel_path, sheet_name="Seed_summary")
    df_filtered = df[df["dataset"] == target_dataset].copy()
    if df_filtered.empty:
        raise ValueError(f"在 dataset 欄位中找不到 '{target_dataset}' 的資料")
        
    return df_filtered

def plot_overview_comparison(df_full, output_dir, target_dataset):
    """將 Single Feature (1個bar) 與其他模型的 4 個 Feature Sets 統整畫在同一張大圖"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 取得所有的 feature sets 列表
    feature_sets = sorted(df_full["feature_set"].unique())
    
    # 定義 X 軸要顯示的項目與對應標籤、顏色
    # 規則：Single Feature 只放 1 個（例如抓 basic 的），其他模型每個 feature set 放 1 個
    x_labels = []
    plot_items = []  # 儲存 (model_name, feature_set, tier_color_key)
    
    # 1. 先加入 Single Feature (Tier 1) - 只取一項
    single_feature_name = "single_feature_feature_60"
    x_labels.append("Single Feature(60)\n(Baseline)")
    plot_items.append((single_feature_name, "basic", "Tier 1 (Baselines)"))
    
    # 2. 加入其他模型 (Tier 2 & Tier 3)，每個 model 對應各自的 4 個 feature sets
    other_models = [
        ("GCN_RF", "Tier 2 (Unsupervised)"),
        ("RF", "Tier 3 (Deep/Graph)")
    ]
    
    for model_name, tier_key in other_models:
        for fs in feature_sets:
            x_labels.append(f"{model_name}\n({fs})")
            plot_items.append((model_name, fs, tier_key))

    # 準備畫布 (1 列, 3 欄)
    fig, axes = plt.subplots(1, 3, figsize=(max(16, len(x_labels) * 0.8), 6))
    fig.suptitle(f"Model & Feature Set Performance Comparison\n(Dataset: {target_dataset})", 
                 fontsize=16, fontweight="bold", y=1.05)

    for i, metric_info in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        metric_base = metric_info["col"]
        mean_col = f"{metric_base}_mean"
        
        means = []
        bar_colors = []
        
        # 依序抓取每個 item 的數據
        for model_name, fs, tier_key in plot_items:
            # 若為 single_feature，強制抓 basic
            if "single_feature" in model_name:
                row = df_full[(df_full["model"] == model_name) & (df_full["feature_set"] == "basic")]
            else:
                row = df_full[(df_full["model"] == model_name) & (df_full["feature_set"] == fs)]
                
            if not row.empty and mean_col in row.columns:
                means.append(row[mean_col].values[0])
            else:
                means.append(0.0)
                
            bar_colors.append(TIER_COLORS[tier_key])

        # 繪製長條圖
        x_pos = np.arange(len(plot_items))
        bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.85, edgecolor="black")
        
        ax.set_title(metric_info["title"], fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0.5, 1.05)  # 確保 Y 軸統一在 0~1 之間
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在每個 Bar 上方標註具體數值
        for bar, mean_val in zip(bars, means):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

    # 建立自訂圖例 (Legend) 來標示 Tier 1, 2, 3
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color, label=tier) 
                      for tier, color in TIER_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11)
    
    plt.tight_layout()
    
    # 存檔
    save_path = output_dir / f"Overview_Comparison_{target_dataset}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved overview comparison plot: {save_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Plot overview comparison of models and feature sets")
    parser.add_argument("--excel", type=str, default="./Analysis/training_results_all.xlsx", help="Path to Excel file")
    parser.add_argument("--output", type=str, default="./Analysis/Plots_Overview", help="Directory to save plots")
    parser.add_argument("--dataset", type=str, default="val", help="Target dataset to plot (e.g., test_after_event, val)")
    args = parser.parse_args()
    
    # 1. 讀取與過濾資料
    df = load_data(args.excel, args.dataset)
    
    # 2. 產出綜合對比圖
    plot_overview_comparison(df, args.output, args.dataset)

if __name__ == "__main__":
    main()