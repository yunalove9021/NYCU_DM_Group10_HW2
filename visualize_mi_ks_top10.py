"""Plot basic MI/KS and five-seed RF/GCN_RF importance Top 10."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from visualize_features import feature_scores, importance_runs, plot_stability, save_figure


ROOT = Path(__file__).resolve().parent


def run_importance(config_path, save_dir, output_dir=None, seeds=None, expected_seeds=5):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(output_dir).resolve() if output_dir is not None else (config_path.parent / config["output"]).resolve()
    seeds = config["random_seeds"] if seeds is None else seeds
    seeds = [seeds] if type(seeds) is int else seeds
    if (expected_seeds < 2 or not isinstance(seeds, list) or len(seeds) != expected_seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError(f"Require {expected_seeds} distinct nonnegative seeds (minimum 2)")
    seeds = sorted(seeds)
    summaries, details, sources = {}, {}, {}
    # Check both model result sets before writing new importance outputs.
    for model in ("RF", "GCN_RF"):
        details[model], summaries[model], sources[model] = importance_runs(output, model, "basic", seeds)
    destination = Path(save_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_stability(summaries, destination, "basic", 10, seeds, filename="basic_importance_top10")
    for model, summary in summaries.items():
        summary.to_csv(destination / f"basic_{model}_importance_all_features.csv", index=False, encoding="utf-8-sig")
        top = summary.head(10).reset_index(drop=True).copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        top.to_csv(destination / f"basic_{model}_importance_top10.csv", index=False, encoding="utf-8-sig")
        details[model].to_csv(destination / f"basic_{model}_importance_by_seed.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "feature_set": "basic", "models": ["RF", "GCN_RF"], "seeds": seeds,
        "seed_count": len(seeds), "output_dir": str(output), "source_files": sources,
        "ranking": "Top 10 by mean impurity importance across seeds; ties sorted by feature name",
        "error_bars": "Sample standard deviation, ddof=1",
        "top10_count": "Number of individual seed rankings containing the feature in Top 10",
        "notes": ["RF and GCN_RF are separate panels with separate feature rankings.",
                  "GCN_RF importance comes from its downstream RF, not GCN parameters.",
                  "Embedding axes are not aligned across seeds; coordinate-wise variation is descriptive.",
                  "Missing seed results or inconsistent feature spaces are errors, not zero importance."],
    }
    (destination / "basic_importance_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"basic RF / GCN_RF importance Top 10 ({len(seeds)} seeds): {destination}", flush=True)
    return destination


def run(config_path, save_dir, random_state=42, discrete_features=()):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_path = config_path.parent / "feature_sets" / "basic.json"
    features = json.loads(feature_path.read_text(encoding="utf-8-sig"))
    label = config["data"].get("label_column", "y")
    if (not isinstance(features, list) or not features
            or any(not isinstance(feature, str) or not feature for feature in features)
            or len(features) != len(set(features)) or label in features):
        raise ValueError("basic.json must contain unique feature names excluding the label")
    if set(discrete_features) - set(features):
        raise ValueError("--discrete-features must refer to features in basic.json")
    train_path = (config_path.parent / config["input"] / config["data"]["train_file"]).resolve()
    columns = pd.read_csv(train_path, nrows=0).columns
    missing = [name for name in [label, *features] if name not in columns]
    if missing:
        raise ValueError(f"Train CSV is missing required columns: {missing}")
    train = pd.read_csv(train_path, usecols=[label, *features])
    if not train[label].dropna().isin([0, 1]).all():
        raise ValueError("Train labels must be 0 (licit), 1 (illicit), or missing")
    if any((train[label] == value).sum() < 2 for value in [0, 1]):
        raise ValueError("Require at least two labeled train nodes of each class")
    print(f"basic: {len(features)} features, {len(train):,} train nodes, "
          f"{int(train[label].notna().sum()):,} labeled", flush=True)
    # Same scoring method as visualize_features.py; no model outputs or seed runs are read.
    scores = feature_scores(train, label, features, random_state, discrete_features)
    top_mi = scores.dropna(subset=["mi_nats"]).sort_values(
        ["mi_nats", "feature"], ascending=[False, True]).head(10)
    top_ks = scores.dropna(subset=["ks_distance"]).sort_values(
        ["ks_distance", "feature"], ascending=[False, True]).head(10)
    if min(len(top_mi), len(top_ks)) < 10:
        raise ValueError("Need at least 10 features with valid MI and KS scores")
    destination = Path(save_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, table, metric, title, color in [
        (axes[0], top_mi, "mi_nats", "Mutual Information (nats)", "#277DA8"),
        (axes[1], top_ks, "ks_distance", "KS distance", "#D88B35"),
    ]:
        selected = table.iloc[::-1]
        bars = ax.barh(selected.feature, selected[metric], color=color)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in selected[metric]], padding=4, fontsize=9)
        ax.set(title=f"Top 10 by {title}", xlabel=title)
        ax.set_xlim(0, max(0.01, float(selected[metric].max())) * 1.22)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("basic: MI and KS Top 10 | labeled train nodes only", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, destination, "basic_mi_ks_top10")
    scores.to_csv(destination / "basic_mi_ks_all_features.csv", index=False, encoding="utf-8-sig")
    for name, table in [("mi", top_mi), ("ks", top_ks)]:
        ranked = table.reset_index(drop=True).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked.to_csv(destination / f"basic_{name}_top10.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "feature_set": "basic", "feature_file": str(feature_path.resolve()),
        "train_file": str(train_path), "label_column": label, "n_features": len(features),
        "n_train_nodes": len(train), "n_licit": int((train[label] == 0).sum()),
        "n_illicit": int((train[label] == 1).sum()), "mi_random_state": random_state,
        "mi_n_neighbors": 3, "mi_discrete_features": list(discrete_features),
        "notes": ["Only the configured train CSV is read; no validation/test data or model outputs.",
                  "Unlabeled nodes excluded. Missing/nonfinite feature values excluded per feature.",
                  "MI uses nats and continuous kNN estimation unless a feature is explicitly discrete.",
                  "KS is the empirical two-class distance D, not a p-value.",
                  "Both rankings are independent and select Top 10 from the full basic feature set."],
    }
    (destination / "basic_mi_ks_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Top 10 figures (PNG/PDF) and ranking CSVs: {destination}", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--feature-set", choices=["basic"], default="basic",
                        help="This independent entry point analyzes basic only")
    parser.add_argument("--save-dir", type=Path,
                        default=ROOT / "Analysis" / "feature_visualization" / "basic_top10")
    parser.add_argument("--random-state", type=int, default=42, help="MI estimator random seed, not a model-training seed")
    parser.add_argument("--discrete-features", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, help="Input RF/GCN_RF training output root; defaults to YAML output")
    parser.add_argument("--seeds", nargs="+", type=int, help="Importance seeds; defaults to YAML random_seeds")
    parser.add_argument("--expected-seeds", type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--importance-only", action="store_true", help="Only generate RF/GCN_RF importance Top 10; do not recompute MI/KS")
    mode.add_argument("--mi-ks-only", action="store_true", help="Only generate train MI/KS; do not require model results")
    args = parser.parse_args(argv)
    try:
        if not args.mi_ks_only:
            run_importance(args.config, args.save_dir, args.output_dir, args.seeds, args.expected_seeds)
        if not args.importance_only:
            run(args.config, args.save_dir, args.random_state, args.discrete_features)
        return args.save_dir.resolve()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
