"""Train-only feature diagnostics and five-seed RF/GCN_RF importance plots."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
import yaml


ROOT = Path(__file__).resolve().parent
FOCUS = ["feature_5", "feature_18", "feature_60", "feature_90"]
COLORS = {0: "#277DA8", 1: "#D85745"}
STRUCTURAL = ["in_degree", "out_degree", "neighbor_mean_degree", "local_clustering"]


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("png", "pdf"):
            # Finish writing before replacing an existing figure (e.g. open in IDE).
            with tempfile.NamedTemporaryFile(dir=directory, suffix=f".{extension}", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                fig.savefig(temporary, format=extension, dpi=220, bbox_inches="tight")
                temporary.replace(directory / f"{name}.{extension}")
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        plt.close(fig)


def feature_scores(data, label, features, random_state=42, discrete=()):
    """Estimate MI and empirical KS distance using known train labels only."""
    rows = []
    for index, feature in enumerate(features):
        x = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        known = data[label].notna() & x.notna()
        values = x.loc[known].to_numpy(dtype=float)
        y = data.loc[known, label].to_numpy(dtype=int)
        normal, anomaly = values[y == 0], values[y == 1]
        score = {"feature": feature, "n_licit": len(normal), "n_illicit": len(anomaly),
                 "n_excluded_labeled": int((data[label].notna() & x.isna()).sum()),
                 "n_unique": len(np.unique(values)), "mi_nats": np.nan, "ks_distance": np.nan,
                 "mi_type": "discrete" if feature in discrete else "continuous"}
        if len(normal) >= 2 and len(anomaly) >= 2:
            score["ks_distance"] = float(ks_2samp(normal, anomaly, method="asymp").statistic)
            score["mi_nats"] = 0.0 if score["n_unique"] <= 1 else float(mutual_info_classif(
                values.reshape(-1, 1), y, discrete_features=feature in discrete,
                n_neighbors=3, random_state=random_state)[0])
        else:
            score["status"] = "insufficient_class_samples"
        rows.append(score)
        if (index + 1) % 25 == 0:
            print(f"MI/KS: {index + 1}/{len(features)} features", flush=True)
    return pd.DataFrame(rows).sort_values(["mi_nats", "feature"], ascending=[False, True])


def importance_runs(output, model, set_name, seeds):
    frames, sources, expected = [], [], None
    for seed in seeds:
        path = output / f"{model}_output" / set_name / "eval" / f"{set_name}_seed_{seed}_{model}_feature_importance.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required seed {seed}: {path}")
        frame = pd.read_csv(path)
        if not {"feature", "importance"} <= set(frame):
            raise ValueError(f"Feature importance CSV requires feature and importance: {path}")
        if frame.empty or frame.feature.isna().any() or frame.feature.duplicated().any():
            raise ValueError(f"Empty/duplicate/missing feature names: {path}")
        values = pd.to_numeric(frame.importance, errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid importances: {path}")
        current = set(frame.feature)
        if expected is not None and current != expected:
            raise ValueError(f"Feature spaces differ across seeds: {model}/{set_name}, seed {seed}")
        expected = current
        frame = frame.assign(importance=values).sort_values(["importance", "feature"], ascending=[False, True])
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["top10"] = frame["rank"] <= 10
        frame["seed"] = seed
        frames.append(frame[["feature", "importance", "rank", "top10", "seed"]])
        sources.append(str(path.resolve()))
    runs = pd.concat(frames, ignore_index=True)
    summary = runs.groupby("feature", as_index=False).agg(
        importance_mean=("importance", "mean"), importance_sd=("importance", "std"),
        top10_count=("top10", "sum"), mean_rank=("rank", "mean"), seed_count=("seed", "nunique"))
    return runs, summary.sort_values(["importance_mean", "feature"], ascending=[False, True]), sources


def plot_rankings(scores, directory, set_name, top):
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, top * 0.37)))
    for ax, metric, title, color in zip(axes, ["mi_nats", "ks_distance"],
                                      ["Mutual information (nats)", "KS distance"], ["#277DA8", "#D88B35"]):
        selected = scores.dropna(subset=[metric]).sort_values([metric, "feature"], ascending=[False, True]).head(top).iloc[::-1]
        ax.barh(selected.feature, selected[metric], color=color)
        ax.set(xlabel=title, title=f"Top {len(selected)} by {title}")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"{set_name}: feature-label association | labeled train nodes only", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, directory, "01_mi_ks_rankings")


def plot_stability(summaries, directory, set_name, top, seeds):
    fig, axes = plt.subplots(1, len(summaries), figsize=(8 * len(summaries), max(6, top * 0.39)), squeeze=False)
    for ax, (model, summary) in zip(axes[0], summaries.items()):
        selected = summary.head(top).iloc[::-1]
        embedding = selected.feature.str.startswith("emb_dim_")
        colors = np.where(embedding, "#D88B35", "#277DA8")
        labels = [f"{feature}  [{int(hits)}/{len(seeds)}]" for feature, hits in zip(selected.feature, selected.top10_count)]
        ax.barh(labels, selected.importance_mean, xerr=selected.importance_sd.fillna(0),
                color=colors, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#333333"})
        ax.axvline(0, color="#777777", linewidth=0.6)
        ax.set(title=f"{model} | top {len(selected)} mean importance",
               xlabel=f"RF impurity importance: mean +/- sample SD (n={len(seeds)})")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"{set_name}: importance stability | [k/{len(seeds)}] = appearances in seed Top 10", fontsize=14)
    fig.legend(handles=[Patch(color="#277DA8", label="Original feature"),
                        Patch(color="#D88B35", label="GCN embedding coordinate")],
               loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2)
    fig.text(0.5, 0.008, "GCN embedding axes are not aligned across seeds; coordinate-wise variation is descriptive.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    save_figure(fig, directory, "03_importance_stability")


def select_correlation_features(scores, rf_summary, available, top, source):
    """Reserve focus features, then fill from the requested ranking(s)."""
    selected = [name for name in FOCUS if name in available]
    mi = scores.sort_values(["mi_nats", "feature"], ascending=[False, True]).feature.tolist()
    ks = scores.sort_values(["ks_distance", "feature"], ascending=[False, True]).feature.tolist()
    rf = [] if rf_summary is None else [name for name in rf_summary.feature if name in available]
    orders = {"mi": [mi], "ks": [ks], "rf": [rf], "combined": [mi, ks, rf]}[source]
    if source == "rf" and not rf:
        raise ValueError("--heatmap-source rf requires RF importance results")
    for index in range(max((len(order) for order in orders), default=0)):
        for order in orders:
            if index < len(order) and order[index] not in selected and order[index] in available:
                selected.append(order[index])
            if len(selected) >= top:
                return selected[:top]
    return selected


def correlation_plot(train, selected, candidates, directory, set_name, threshold):
    clean = train[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        raise ValueError(f"Insufficient complete train rows for Spearman: {set_name}")
    usable = [column for column in selected if clean[column].nunique() > 1]
    if len(usable) < 2:
        raise ValueError(f"Need at least two nonconstant train features for heatmap: {set_name}")
    corr = clean[usable].corr(method="spearman")
    distances = np.clip(1 - corr.abs().to_numpy(), 0, 1)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(tree)
    ordered = corr.iloc[order, order]
    ordered.to_csv(directory / "spearman_correlation.csv", encoding="utf-8-sig")
    pairs = [{"feature_a": a, "feature_b": b, "spearman_rho": float(corr.loc[a, b]),
              "abs_rho": float(abs(corr.loc[a, b])), "high_correlation": bool(abs(corr.loc[a, b]) >= threshold)}
             for i, a in enumerate(usable) for b in usable[i + 1:]]
    pd.DataFrame(pairs).sort_values("abs_rho", ascending=False).to_csv(directory / "spearman_pairs.csv", index=False)
    fig = plt.figure(figsize=(13, 13))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 0.035], height_ratios=[0.18, 1], wspace=0.08, hspace=0.02)
    ax_tree, ax, ax_color = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=0, above_threshold_color="#555555")
    ax_tree.axis("off")
    chart = ax.imshow(ordered, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [name + (" *" if name not in candidates else "") for name in ordered.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for tick, name in zip(ax.get_xticklabels(), ordered.columns):
        if name in FOCUS:
            tick.set_color("#B32E25")
            tick.set_fontweight("bold")
    for tick, name in zip(ax.get_yticklabels(), ordered.columns):
        if name in FOCUS:
            tick.set_color("#B32E25")
            tick.set_fontweight("bold")
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            value = ordered.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.6 else "#222222")
    fig.colorbar(chart, cax=ax_color, label="Spearman rho")
    fig.suptitle(f"{set_name}: train-only Spearman correlation (n={len(clean):,})\n"
                 "Average-linkage clustering on 1 - |rho|; red labels = focus features", y=0.98)
    fig.text(0.1, 0.015, "* = focus reference outside this candidate set; no validation/test nodes used", fontsize=9)
    fig.subplots_adjust(bottom=0.17, top=0.92, left=0.14, right=0.94)
    save_figure(fig, directory, "02_spearman_clustered")
    return {"features": ordered.columns.tolist(), "n_complete_train_rows": len(clean),
            "excluded_constant_features": [name for name in selected if name not in usable],
            "high_correlation_pairs": [pair for pair in pairs if pair["high_correlation"]]}


def graph_features(train, edges):
    """Topology only: directed degrees, undirected neighbor degree and clustering."""
    if not {"txId", "time_step"} <= set(train):
        raise ValueError("Graph analysis requires train txId and time_step")
    nodes = train.loc[train.time_step.between(1, 30)].copy()
    if nodes.empty or nodes.txId.isna().any() or nodes.txId.duplicated().any():
        raise ValueError("Train t1-30 requires nonempty, unique, nonmissing node IDs")
    if not {"txId1", "txId2"} <= set(edges):
        raise ValueError("Train edges require txId1 and txId2")
    count_before = len(edges)
    if "time_step" in edges:
        edges = edges.loc[edges.time_step.between(1, 30)]
    ids = pd.Index(nodes.txId)
    source, target = ids.get_indexer(edges.txId1), ids.get_indexer(edges.txId2)
    if (source < 0).any() or (target < 0).any():
        raise ValueError("Train edge endpoint missing from train t1-30 nodes")
    pairs = np.unique(np.column_stack([source, target]), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(pairs.tolist())
    neighbors = nx.average_neighbor_degree(graph)
    clustering = nx.clustering(graph)
    features = pd.DataFrame({
        "in_degree": np.bincount(pairs[:, 1], minlength=len(nodes)),
        "out_degree": np.bincount(pairs[:, 0], minlength=len(nodes)),
        "neighbor_mean_degree": [neighbors[i] for i in range(len(nodes))],
        "local_clustering": [clustering[i] for i in range(len(nodes))],
    }, index=nodes.index)
    return pd.concat([nodes[[column for column in ("txId", "time_step", "y") if column in nodes]], features], axis=1), {
        "n_nodes": len(nodes), "n_edges_read": count_before, "n_directed_edges_used": len(pairs),
        "n_undirected_edges": graph.number_of_edges(),
        "definition": "Unique edges, self-loops removed. Directed in/out degree; neighbor mean degree and clustering on undirected projection. Isolates have 0.",
    }


def plot_graph_distributions(graph_data, label, scores, directory):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    summary = []
    for ax, feature in zip(axes.flat, STRUCTURAL):
        for value, name in [(0, "Licit"), (1, "Illicit")]:
            values = graph_data.loc[graph_data[label] == value, feature].to_numpy()
            unique, counts = np.unique(values, return_counts=True)
            ax.step(np.r_[0, unique], np.r_[0, np.cumsum(counts) / len(values)], where="post", color=COLORS[value],
                    linewidth=1.8, label=f"{name} (n={len(values):,})")
            summary.append({"feature": feature, "class": name, "n": len(values), "mean": values.mean(),
                            "median": np.median(values), "q25": np.quantile(values, 0.25),
                            "q75": np.quantile(values, 0.75), "fraction_zero": float((values == 0).mean())})
        score = scores.set_index("feature").loc[feature]
        ax.set(title=f"{feature}\nKS D={score.ks_distance:.3f} | MI={score.mi_nats:.4f} nats",
               xlabel=feature, ylabel="Empirical cumulative probability", ylim=(-0.02, 1.02))
        if feature != "local_clustering":
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f"{feature} (symlog scale; linear near 0)")
        ax.set_xlim(0, 1 if feature == "local_clustering" else max(1, graph_data[feature].max() * 1.08))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    pd.DataFrame(summary).to_csv(directory / "graph_class_summary.csv", index=False)
    fig.suptitle("Graph-structural feature distributions | train t1-30 only\nTopology uses all train nodes; class comparisons use labeled nodes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, directory, "04_graph_structural_ecdf")


def run(args):
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve() if args.output_dir else (config_path.parent / config["output"]).resolve()
    models = list(dict.fromkeys(args.models))
    input_dir = (config_path.parent / config["input"]).resolve()
    train_path = input_dir / config["data"]["train_file"]
    edge_path = input_dir / config["data"]["edge_files"]["train"] if "GCN_RF" in models else None
    seeds = args.seeds if args.seeds is not None else config["random_seeds"]
    seeds = [seeds] if isinstance(seeds, int) else seeds
    if (not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != args.expected_seeds or len(set(seeds)) != len(seeds)):
        raise ValueError(f"Require {args.expected_seeds} distinct seeds, got {seeds}; pass --seeds explicitly if needed")
    seeds = sorted(seeds)
    sets = args.feature_sets or sorted({path.name for model in models for path in (output / f"{model}_output").glob("*") if path.is_dir()})
    if not sets:
        raise ValueError(f"No RF/GCN_RF feature-set output folders found under {output}")
    all_results, sources = {}, []
    # Validate every seed before starting any expensive calculation or plotting.
    for set_name in sets:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", set_name):
            raise ValueError(f"Invalid feature set name: {set_name}")
        all_results[set_name] = {}
        for model in models:
            runs, summary, files = importance_runs(output, model, set_name, seeds)
            all_results[set_name][model] = (runs, summary)
            sources.extend(files)
    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    label = config["data"].get("label_column", "y")
    if label not in header:
        raise ValueError(f"Train CSV missing label column {label}")
    candidates = {set_name: sorted({feature for _, summary in results.values() for feature in summary.feature
                                   if feature in header and feature not in {label, "txId", "time_step", "raw_class"}})
                  for set_name, results in all_results.items()}
    if any(not features for features in candidates.values()):
        raise ValueError("Every feature set must contain original features present in train CSV")
    columns = sorted({feature for features in candidates.values() for feature in features} | (set(FOCUS) & set(header)))
    metadata_columns = [name for name in ("txId", "time_step", label) if name in header]
    train = pd.read_csv(train_path, usecols=list(dict.fromkeys(metadata_columns + columns)))
    if not train[label].dropna().isin([0, 1]).all() or set(train[label].dropna().unique()) != {0, 1}:
        raise ValueError("Train labels must contain both 0 (licit) and 1 (illicit), with optional missing labels")
    directory = args.save_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Read train only: {len(train):,} nodes, {int(train[label].notna().sum()):,} labeled; {len(columns)} candidate/reference features", flush=True)
    scores = feature_scores(train, label, columns, args.random_state, args.discrete_features)
    scores.to_csv(directory / "all_train_mi_ks.csv", index=False, encoding="utf-8-sig")
    details = {}
    for set_name, results in all_results.items():
        dest = directory / set_name
        dest.mkdir(parents=True, exist_ok=True)
        group_scores = scores.loc[scores.feature.isin(candidates[set_name])]
        group_scores.to_csv(dest / "mi_ks_scores.csv", index=False, encoding="utf-8-sig")
        summaries = {model: summary for model, (_, summary) in results.items()}
        for model, (runs, summary) in results.items():
            runs.to_csv(dest / f"{model}_importance_by_seed.csv", index=False)
            summary.to_csv(dest / f"{model}_importance_stability.csv", index=False)
        plot_rankings(group_scores, dest, set_name, args.top)
        selected = select_correlation_features(group_scores, summaries.get("RF"), columns, args.top, args.heatmap_source)
        details[set_name] = correlation_plot(train, selected, candidates[set_name], dest, set_name, args.correlation_threshold)
        plot_stability(summaries, dest, set_name, args.top, seeds)
        print(f"Saved candidate rankings, clustered Spearman and seed stability: {dest}", flush=True)
    focused = [name for name in FOCUS if name in columns]
    focus_corr = train[focused].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    focus_corr.to_csv(directory / "focus_spearman.csv", encoding="utf-8-sig")
    if "GCN_RF" in models:
        print("Computing graph-structural features on train t1-30...", flush=True)
        graph_input = train.rename(columns={label: "y"}) if label != "y" else train
        graph_data, graph_info = graph_features(graph_input, pd.read_csv(edge_path))
        if (graph_data.y == 0).sum() < 2 or (graph_data.y == 1).sum() < 2:
            raise ValueError("Graph train t1-30 needs at least two labeled nodes in each class")
        graph_scores = feature_scores(graph_data, "y", STRUCTURAL, args.random_state, ["in_degree", "out_degree"])
        graph_data.to_csv(directory / "graph_structural_features_train.csv", index=False, encoding="utf-8-sig")
        graph_scores.to_csv(directory / "graph_mi_ks_scores.csv", index=False)
        plot_graph_distributions(graph_data, "y", graph_scores, directory)
    else:
        graph_info = {"status": "not_requested_without_GCN_RF"}
    manifest = {
        "train_file": str(train_path.resolve()), "train_edge_file": str(edge_path.resolve()) if edge_path else None,
        "training_output": str(output), "models": models, "seeds": seeds,
        "n_train_nodes": len(train), "n_train_labeled": int(train[label].notna().sum()),
        "n_train_licit": int((train[label] == 0).sum()), "n_train_illicit": int((train[label] == 1).sum()),
        "feature_importance_files": sources, "top": args.top, "heatmap_source": args.heatmap_source,
        "mi_random_state": args.random_state, "mi_discrete_features": args.discrete_features,
        "library_versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "networkx"]},
        "correlation_threshold": args.correlation_threshold, "feature_sets": details,
        "focus_missing_from_train": [name for name in FOCUS if name not in focused],
        "focus_high_correlation_pairs": [{"a": a, "b": b, "rho": float(focus_corr.loc[a, b])}
                                         for i, a in enumerate(focused) for b in focused[i + 1:]
                                         if abs(focus_corr.loc[a, b]) >= args.correlation_threshold],
        "graph": graph_info,
        "methods": {
            "mi": "MI(feature; binary train label), nats; continuous kNN n_neighbors=3 unless --discrete-features; constant MI=0; original class proportions, no resampling",
            "ks": "Two-sided empirical KS D; no p-values/inference, including for discrete graph degrees",
            "spearman": "All train nodes, including unlabeled. Heatmap uses complete rows; average linkage on 1-|rho|. Focus references may be outside candidate set.",
            "importance": "RF mean decrease in impurity in each panel; GCN_RF importance is from its downstream RF, not GCN weights. SD uses ddof=1. No cross-model feature averaging.",
            "embedding_caveat": "Embedding coordinate identities may rotate/permutate across independently trained seed encoders; coordinate-wise stability is descriptive, not semantic alignment.",
            "graph": "Topology built without labels from all train t1-30 nodes; labels used only for ECDF/MI/KS. No transaction amount semantics inferred from feature_*.",
        },
    }
    (directory / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Figures (PNG/PDF), CSV tables and analysis_manifest.json: {directory}", flush=True)
    return directory


def model_type(value):
    value = value.replace("-", "_")
    if value not in {"RF", "GCN_RF"}:
        raise argparse.ArgumentTypeError(f"Model '{value}' is not supported; use RF or GCN_RF")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output-dir", type=Path, help="Input training output root; defaults to YAML output")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "Analysis" / "feature_visualization")
    parser.add_argument("--models", nargs="+", type=model_type, default=["RF", "GCN_RF"])
    parser.add_argument("--feature-sets", nargs="+", help="Output feature-set folder names; defaults to all discovered sets")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit seed IDs; defaults to YAML random_seeds")
    parser.add_argument("--expected-seeds", type=int, default=5, help="Require this many complete seed results (default: 5)")
    parser.add_argument("--top", type=int, choices=range(15, 21), default=20, metavar="15..20")
    parser.add_argument("--heatmap-source", choices=["combined", "mi", "ks", "rf"], default="combined")
    parser.add_argument("--correlation-threshold", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42, help="MI estimator random seed")
    parser.add_argument("--discrete-features", nargs="*", default=[], help="Original candidate features to treat as discrete in MI")
    args = parser.parse_args(argv)
    if args.expected_seeds < 2 or not 0 <= args.correlation_threshold <= 1:
        parser.error("Require expected-seeds >= 2 and correlation-threshold in [0, 1]")
    try:
        return run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
