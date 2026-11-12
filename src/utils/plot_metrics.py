"""Visualize model performances across tiers and feature sets from Excel summary."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. 依照使用者需求，手動設定 Tier 1, 2, 3 的模型名稱
#    (名稱必須與 Excel 中「Seed_summary」分頁的 model 欄位字串完全一致)
# ==============================================================================
TIER_GROUPS = {
    "Tier 1 (Baselines)": ["single_feature_feature_60"],
    "Tier 2 (Unsupervised)": ["GCN_RF"],
    "Tier 3 (Deep/Graph)": ["RF"]
}

# 設定各 Tier 的對應顏色 (Matplotlib 色碼)
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
    """讀取 Excel 總表，並過濾出目標資料集 (例如 test_after_event)"""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 檔案: {excel_path}")
        
    df = pd.read_excel(excel_path, sheet_name="Seed_summary")
    
    # 篩選我們想看畫圖的資料集
    df_filtered = df[df["dataset"] == target_dataset].copy()
    if df_filtered.empty:
        raise ValueError(f"在 dataset 欄位中找不到 '{target_dataset}' 的資料")
        
    return df_filtered

def plot_feature_set(df_full, feature_set_name, output_dir, target_dataset):
    """為單一 Feature Set 繪製包含 3 個指標的子圖"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備畫布 (1 列, 3 欄)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Model Performance Comparison - Feature Set: {feature_set_name}\n(Dataset: {target_dataset})", 
                 fontsize=16, fontweight="bold", y=1.05)

    # 彙整需要畫出來的模型清單與對應的顏色
    plot_models = []
    plot_colors = []
    
    for tier_name, models in TIER_GROUPS.items():
        for model in models:
            plot_models.append(model)
            plot_colors.append(TIER_COLORS[tier_name])

    for i, metric_info in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        metric_base = metric_info["col"]
        mean_col = f"{metric_base}_mean"
        std_col = f"{metric_base}_std"
        
        means = []
        stds = []
        
        # 依序抓取每個模型在這個 Feature Set 的數據
        for model in plot_models:
            # 判斷邏輯：如果是 single_feature 且當前不是 basic，就強制去抓 basic 的數據
            if "single_feature" in model and feature_set_name != "basic":
                row = df_full[(df_full["model"] == model) & (df_full["feature_set"] == "basic")]
            else:
                row = df_full[(df_full["model"] == model) & (df_full["feature_set"] == feature_set_name)]

            if not row.empty and mean_col in row.columns:
                means.append(row[mean_col].values[0])
                std_val = row[std_col].values[0]
                stds.append(0.0 if pd.isna(std_val) else std_val)
            else:
                means.append(0.0)
                stds.append(0.0)
                
        # 繪製長條圖
        x_pos = np.arange(len(plot_models))
        bars = ax.bar(x_pos, means, color=plot_colors, 
                      alpha=0.85, edgecolor="black")
        
        ax.set_title(metric_info["title"], fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_models, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)  # 確保 Y 軸統一在 0~1 之間 (預留標籤空間)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在每個 Bar 上方標註具體數值
        for bar, mean_val in zip(bars, means):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

    # 建立自訂圖例 (Legend) 來標示 Tier 1, 2, 3
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color, label=tier) 
                      for tier, color in TIER_COLORS.items()]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)
    
    plt.tight_layout()
    
    # 存檔
    save_path = output_dir / f"Performance_{feature_set_name}_{target_dataset}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Plot model performances by Tiers")
    parser.add_argument("--excel", type=str, default="./Analysis/training_results_all.xlsx", help="Path to the training results Excel file")
    parser.add_argument("--output", type=str, default="./Analysis/Plots", help="Directory to save the plots")
    parser.add_argument("--dataset", type=str, default="val", help="Target dataset to plot (e.g., test_after_event, val)")
    args = parser.parse_args()
    
    # 1. 讀取與過濾資料 (只依 dataset 過濾，保留完整 feature_set 供交叉引用)
    df = load_data(args.excel, args.dataset)
    
    # 2. 取得所有的 feature sets
    feature_sets = df["feature_set"].unique()
    print(f"Found Feature Sets: {list(feature_sets)}")
    
    # 3. 為每個 feature set 畫一張大圖
    for fs in feature_sets:
        plot_feature_set(df, fs, args.output, args.dataset)

if __name__ == "__main__":
    main()