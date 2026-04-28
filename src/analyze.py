#!/usr/bin/env python3
"""
Analysis and plotting for LLM probability calibration experiment.

Reads saved results (results.jsonl) and generates:
1. Entropy histograms (P and C distributions)
2. Calibration curves (reliability diagrams) for Tier 1/2
3. Confidence vs entropy scatter plots
4. Modality analysis (distribution of modes)
5. Attractor analysis (clustering at round numbers)
6. Order effect analysis (KL divergence between conditions A and B)
7. Cross-model comparison (when multiple models found)

Usage:
    python src/analyze.py --results results/Qwen-Qwen3-8B/ --output reports/figures/qwen/ --no-show
    python src/analyze.py --results results/ --output reports/figures/comparison/ --no-show
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # Must be before pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))
from logit_extractor import compute_entropy, kl_divergence, count_modes


def number_to_token_label(n: int) -> str:
    """Format integer 0-100 as its token sequence, e.g. 7 -> '[7]', 57 -> '[5][7]'."""
    return "".join(f"[{c}]" for c in str(n))


def find_results_files(results_dir: str) -> dict:
    """Returns {model_name: path_to_results_jsonl}.

    Auto-detects whether the given path contains results.jsonl directly
    OR contains model subdirectories with results.jsonl inside them.
    """
    p = Path(results_dir)
    # Direct: results_dir/results.jsonl
    if (p / "results.jsonl").exists():
        return {"default": p / "results.jsonl"}
    # Subdirs: results_dir/*/results.jsonl
    found = {}
    for subdir in sorted(p.iterdir()):
        if subdir.is_dir() and (subdir / "results.jsonl").exists():
            found[subdir.name] = subdir / "results.jsonl"
    return found


def load_results(results_path: Path) -> pd.DataFrame:
    """Load results from a single JSONL file into a DataFrame."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    records = []
    with open(results_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Convert list distributions back to numpy arrays
                for key in list(entry.keys()):
                    if key.endswith("_dist_A") or key.endswith("_dist_B"):
                        entry[key] = np.array(entry[key])
                records.append(entry)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    print(f"[+] Loaded {len(df)} results from {results_path}")
    print(f"    Tier breakdown: {dict(df['tier'].value_counts().sort_index())}")
    return df


def plot_entropy_histograms(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Plot entropy histograms for P and C distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = "Entropy Distributions of Logit-Level Probability Estimates"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    # Condition A
    axes[0, 0].hist(df["p_entropy_A"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("P entropy (Condition A: P-first)")
    axes[0, 0].set_xlabel("Shannon Entropy (bits)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(df["p_entropy_A"].median(), color="red", linestyle="--",
                        label=f"Median: {df['p_entropy_A'].median():.2f}")
    axes[0, 0].legend()

    axes[0, 1].hist(df["c_entropy_A"], bins=50, alpha=0.7, color="coral", edgecolor="white")
    axes[0, 1].set_title("C entropy (Condition A: P-first)")
    axes[0, 1].set_xlabel("Shannon Entropy (bits)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].axvline(df["c_entropy_A"].median(), color="red", linestyle="--",
                        label=f"Median: {df['c_entropy_A'].median():.2f}")
    axes[0, 1].legend()

    # Condition B
    axes[1, 0].hist(df["p_entropy_B"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[1, 0].set_title("P entropy (Condition B: C-first)")
    axes[1, 0].set_xlabel("Shannon Entropy (bits)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].axvline(df["p_entropy_B"].median(), color="red", linestyle="--",
                        label=f"Median: {df['p_entropy_B'].median():.2f}")
    axes[1, 0].legend()

    axes[1, 1].hist(df["c_entropy_B"], bins=50, alpha=0.7, color="coral", edgecolor="white")
    axes[1, 1].set_title("C entropy (Condition B: C-first)")
    axes[1, 1].set_xlabel("Shannon Entropy (bits)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].axvline(df["c_entropy_B"].median(), color="red", linestyle="--",
                        label=f"Median: {df['c_entropy_B'].median():.2f}")
    axes[1, 1].legend()

    plt.tight_layout()
    path = Path(output_dir) / "entropy_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_calibration_curves(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Plot reliability diagrams for Tier 1/2 questions."""
    tier12 = df[df["tier"].isin([1, 2])].copy()
    if len(tier12) == 0:
        print("  [!] No Tier 1/2 questions for calibration curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = "Calibration Curves (Reliability Diagrams) — Tier 1 & 2"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, condition, label in [
        (axes[0], "A", "Condition A (P-first)"),
        (axes[1], "B", "Condition B (C-first)"),
    ]:
        predicted = tier12[f"p_argmax_{condition}"].values
        actual = tier12["ground_truth"].values

        # Bin predictions into buckets
        n_bins = 10
        bin_edges = np.linspace(0, 100, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_means = []
        bin_gt_means = []
        bin_counts = []

        for i in range(n_bins):
            mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(predicted[mask].mean())
                bin_gt_means.append(actual[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_means.append(bin_centers[i])
                bin_gt_means.append(np.nan)
                bin_counts.append(0)

        bin_means = np.array(bin_means)
        bin_gt_means = np.array(bin_gt_means)
        bin_counts = np.array(bin_counts)

        # Plot
        valid = ~np.isnan(bin_gt_means)
        ax.bar(bin_means[valid], bin_counts[valid] / bin_counts[valid].max() * 80,
               width=8, alpha=0.2, color="steelblue", label="Sample density")
        ax.plot(bin_means[valid], bin_gt_means[valid], "o-", color="steelblue",
                markersize=8, linewidth=2, label="Model calibration")
        ax.plot([0, 100], [0, 100], "--", color="gray", linewidth=1, label="Perfect calibration")

        # ECE
        ece = np.nansum(bin_counts * np.abs(np.nan_to_num(bin_gt_means) - bin_means)) / bin_counts.sum()
        ax.text(5, 90, f"ECE = {ece:.1f}", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Predicted Probability (%)")
        ax.set_ylabel("Ground Truth Probability (%)")
        ax.set_title(label)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "calibration_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_confidence_vs_entropy(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Scatter plot: verbalized confidence (C argmax) vs P entropy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = "Verbalized Confidence vs. Probability Entropy"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, condition, label in [
        (axes[0], "A", "Condition A (P-first)"),
        (axes[1], "B", "Condition B (C-first)"),
    ]:
        x = df[f"c_argmax_{condition}"]
        y = df[f"p_entropy_{condition}"]

        # Color by tier
        colors = df["tier"].map({1: "steelblue", 2: "coral", 3: "seagreen"})
        ax.scatter(x, y, c=colors, alpha=0.5, s=20)

        # Add correlation
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 2:
            corr = np.corrcoef(x[valid], y[valid])[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=12, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Verbalized Confidence (C argmax)")
        ax.set_ylabel("P Entropy (bits)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Tier 1: Bayesian"),
        Patch(facecolor="coral", label="Tier 2: Classical"),
        Patch(facecolor="seagreen", label="Tier 3: Epistemic"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = Path(output_dir) / "confidence_vs_entropy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_modality_analysis(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Distribution of number of modes in P distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = "Distribution Modality Analysis"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, condition, label in [
        (axes[0], "A", "Condition A (P-first)"),
        (axes[1], "B", "Condition B (C-first)"),
    ]:
        modes = df[f"n_modes_p_{condition}"]
        mode_counts = Counter(modes)
        max_mode = max(mode_counts.keys()) if mode_counts else 5

        x_vals = range(1, max_mode + 1)
        y_vals = [mode_counts.get(m, 0) for m in x_vals]

        ax.bar(x_vals, y_vals, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Number of Modes")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.set_xticks(range(1, max_mode + 1))

        mean_modes = modes.mean()
        ax.axvline(mean_modes, color="red", linestyle="--",
                    label=f"Mean: {mean_modes:.2f}")
        ax.legend()

    plt.tight_layout()
    path = Path(output_dir) / "modality_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_attractor_analysis(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Histogram of P argmax values to detect attractor patterns."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    title = "Attractor Analysis: Distribution of Predicted Probabilities"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, condition, label in [
        (axes[0], "A", "Condition A (P-first)"),
        (axes[1], "B", "Condition B (C-first)"),
    ]:
        values = df[f"p_argmax_{condition}"]

        # Full histogram
        bins = np.arange(-0.5, 101.5, 1)
        counts, _, bars = ax.hist(values, bins=bins, color="steelblue",
                                   edgecolor="white", alpha=0.8)

        # Highlight multiples of 10
        for bar, left_edge in zip(bars, bins[:-1]):
            val = int(left_edge + 0.5)
            if val % 10 == 0 and 0 <= val <= 100:
                bar.set_facecolor("coral")
                bar.set_alpha(1.0)
            elif val % 5 == 0 and 0 <= val <= 100:
                bar.set_facecolor("gold")
                bar.set_alpha(0.9)

        # Compute round-number concentration
        total = len(values)
        mult_10 = sum(1 for v in values if v % 10 == 0) / total * 100 if total > 0 else 0
        mult_5 = sum(1 for v in values if v % 5 == 0) / total * 100 if total > 0 else 0

        ax.text(0.98, 0.95,
                f"% at multiples of 10: {mult_10:.1f}%\n"
                f"% at multiples of 5: {mult_5:.1f}%\n"
                f"(Random expectation: 10.9%, 20.8%)",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Token-format X-axis ticks (every 5 values for attractor plot)
        tick_step = 5
        tick_positions = np.arange(0, 101, tick_step)
        tick_labels = [number_to_token_label(int(n)) for n in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

        ax.set_xlabel("Token sequence (P argmax)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.set_xlim(-1, 101)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="coral", label="Multiple of 10"),
            Patch(facecolor="gold", label="Multiple of 5"),
            Patch(facecolor="steelblue", label="Other"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    path = Path(output_dir) / "attractor_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_order_effects(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Distribution of KL divergence between conditions A and B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = "Order Effects: KL Divergence Between Conditions"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, metric, label in [
        (axes[0], "order_effect_p", "P Distribution Shift"),
        (axes[1], "order_effect_c", "C Distribution Shift"),
    ]:
        values = df[metric].dropna()
        # Clip extreme values for visualization
        clipped = np.clip(values, 0, np.percentile(values, 95))

        ax.hist(clipped, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel("KL(A || B) (nats)")
        ax.set_ylabel("Count")
        ax.set_title(label)

        median_val = values.median()
        mean_val = values.mean()
        ax.axvline(median_val, color="red", linestyle="--",
                    label=f"Median: {median_val:.4f}")
        ax.axvline(mean_val, color="orange", linestyle="--",
                    label=f"Mean: {mean_val:.4f}")
        ax.legend()

        # Per-tier breakdown
        for tier in sorted(df["tier"].unique()):
            tier_vals = df[df["tier"] == tier][metric].dropna()
            tier_label = {1: "Bayesian", 2: "Classical", 3: "Epistemic"}.get(tier, str(tier))
            print(f"    {label} — Tier {tier} ({tier_label}): "
                  f"median={tier_vals.median():.4f}, mean={tier_vals.mean():.4f}")

    plt.tight_layout()
    path = Path(output_dir) / "order_effects.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_error_by_tier(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Box plot of prediction errors by tier and category."""
    tier12 = df[df["tier"].isin([1, 2])].copy()
    if len(tier12) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = "Prediction Error by Tier and Category"
    if model_name:
        title += f"\n({model_name})"
    fig.suptitle(title, fontsize=14)

    for ax, condition, label in [
        (axes[0], "A", "Condition A (P-first)"),
        (axes[1], "B", "Condition B (C-first)"),
    ]:
        error_col = f"p_error_{condition}"
        if error_col not in tier12.columns:
            continue

        data = tier12[["category", error_col]].dropna()
        categories = sorted(data["category"].unique())

        box_data = [data[data["category"] == cat][error_col].values for cat in categories]

        bp = ax.boxplot(box_data, labels=categories, patch_artist=True)
        colors = sns.color_palette("Set2", len(categories))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Category")
        ax.set_ylabel("|Predicted - Ground Truth|")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "error_by_category.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_dir: str, model_name: str = "") -> dict:
    """Generate and save summary statistics."""
    stats = {
        "model": model_name,
        "total_questions": len(df),
        "tier_counts": dict(df["tier"].value_counts().sort_index()),
        "category_counts": dict(df["category"].value_counts()),
    }

    # Entropy stats
    for condition in ["A", "B"]:
        for metric in ["p_entropy", "c_entropy"]:
            col = f"{metric}_{condition}"
            if col in df.columns:
                vals = df[col].dropna()
                stats[f"{col}_mean"] = float(vals.mean())
                stats[f"{col}_median"] = float(vals.median())
                stats[f"{col}_std"] = float(vals.std())

    # Modality stats
    for condition in ["A", "B"]:
        col = f"n_modes_p_{condition}"
        if col in df.columns:
            vals = df[col].dropna()
            stats[f"{col}_mean"] = float(vals.mean())
            stats[f"{col}_median"] = float(vals.median())
            stats[f"{col}_std"] = float(vals.std())
        col_c = f"n_modes_c_{condition}"
        if col_c in df.columns:
            vals = df[col_c].dropna()
            stats[f"{col_c}_mean"] = float(vals.mean())
            stats[f"{col_c}_median"] = float(vals.median())

    # Argmax stats
    for condition in ["A", "B"]:
        col = f"p_argmax_{condition}"
        if col in df.columns:
            vals = df[col]
            stats[f"{col}_mean"] = float(vals.mean())
            stats[f"{col}_pct_mult10"] = float((vals % 10 == 0).mean() * 100)
            stats[f"{col}_pct_mult5"] = float((vals % 5 == 0).mean() * 100)

    # Error stats (Tier 1/2)
    tier12 = df[df["tier"].isin([1, 2])]
    for condition in ["A", "B"]:
        col = f"p_error_{condition}"
        if col in tier12.columns:
            vals = tier12[col].dropna()
            if len(vals) > 0:
                stats[f"{col}_mean"] = float(vals.mean())
                stats[f"{col}_median"] = float(vals.median())
                stats[f"{col}_std"] = float(vals.std())

    # Error by category
    for condition in ["A", "B"]:
        col = f"p_error_{condition}"
        if col in tier12.columns:
            for cat in sorted(tier12["category"].unique()):
                cat_vals = tier12[tier12["category"] == cat][col].dropna()
                if len(cat_vals) > 0:
                    stats[f"{col}_{cat}_mean"] = float(cat_vals.mean())
                    stats[f"{col}_{cat}_median"] = float(cat_vals.median())

    # Order effects
    for metric in ["order_effect_p", "order_effect_c"]:
        if metric in df.columns:
            vals = df[metric].dropna()
            stats[f"{metric}_mean"] = float(vals.mean())
            stats[f"{metric}_median"] = float(vals.median())
            stats[f"{metric}_std"] = float(vals.std())

    suffix = f"_{model_name}" if model_name and model_name != "default" else ""
    path = Path(output_dir) / f"summary_stats{suffix}.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj

    stats = convert_numpy(stats)

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {path}")

    return stats


def print_summary(stats: dict):
    """Print a human-readable summary."""
    model = stats.get("model", "Unknown")
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY — {model}")
    print("=" * 60)

    print(f"\nTotal questions: {stats['total_questions']}")
    print(f"Tier counts: {stats.get('tier_counts', 'N/A')}")

    print("\n--- Entropy ---")
    for cond in ["A", "B"]:
        p_ent = stats.get(f"p_entropy_{cond}_mean", "N/A")
        p_std = stats.get(f"p_entropy_{cond}_std", "N/A")
        c_ent = stats.get(f"c_entropy_{cond}_mean", "N/A")
        c_std = stats.get(f"c_entropy_{cond}_std", "N/A")
        p_med = stats.get(f"p_entropy_{cond}_median", "N/A")
        c_med = stats.get(f"c_entropy_{cond}_median", "N/A")
        label = "P-first" if cond == "A" else "C-first"
        if isinstance(p_ent, float):
            print(f"  Condition {cond} ({label}):")
            print(f"    P entropy: mean={p_ent:.4f}, median={p_med:.4f}, std={p_std:.4f}")
            print(f"    C entropy: mean={c_ent:.4f}, median={c_med:.4f}, std={c_std:.4f}")

    print("\n--- Modality ---")
    for cond in ["A", "B"]:
        col = f"n_modes_p_{cond}"
        m = stats.get(f"{col}_mean", "N/A")
        label = "P-first" if cond == "A" else "C-first"
        if isinstance(m, float):
            print(f"  Condition {cond} ({label}): mean modes = {m:.2f}")

    print("\n--- Attractor Analysis ---")
    for cond in ["A", "B"]:
        pct10 = stats.get(f"p_argmax_{cond}_pct_mult10", "N/A")
        pct5 = stats.get(f"p_argmax_{cond}_pct_mult5", "N/A")
        if isinstance(pct10, float):
            print(f"  Condition {cond}: {pct10:.1f}% at multiples of 10, {pct5:.1f}% at multiples of 5")

    print("\n--- Calibration Error (Tier 1/2) ---")
    for cond in ["A", "B"]:
        err = stats.get(f"p_error_{cond}_mean", "N/A")
        err_std = stats.get(f"p_error_{cond}_std", "N/A")
        err_med = stats.get(f"p_error_{cond}_median", "N/A")
        if isinstance(err, float):
            print(f"  Condition {cond}: MAE = {err:.2f} (median={err_med:.2f}, std={err_std:.2f})")

    print("\n--- Order Effects ---")
    p_oe = stats.get("order_effect_p_mean", "N/A")
    p_oe_med = stats.get("order_effect_p_median", "N/A")
    c_oe = stats.get("order_effect_c_mean", "N/A")
    c_oe_med = stats.get("order_effect_c_median", "N/A")
    if isinstance(p_oe, float):
        print(f"  P distribution shift: mean KL = {p_oe:.4f}, median = {p_oe_med:.4f}")
        print(f"  C distribution shift: mean KL = {c_oe:.4f}, median = {c_oe_med:.4f}")

    print("\n" + "=" * 60)


# =================== CROSS-MODEL COMPARISON PLOTS ===================

def plot_comparison_entropy(all_stats: dict, output_dir: str):
    """Side-by-side entropy comparison across models."""
    models = list(all_stats.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Model Entropy Comparison", fontsize=14)

    x = np.arange(n_models)
    width = 0.35

    for ax, metric_type, title in [
        (axes[0], "p_entropy", "P (Probability) Entropy"),
        (axes[1], "c_entropy", "C (Confidence) Entropy"),
    ]:
        means_A = [all_stats[m].get(f"{metric_type}_A_mean", 0) for m in models]
        stds_A = [all_stats[m].get(f"{metric_type}_A_std", 0) for m in models]
        means_B = [all_stats[m].get(f"{metric_type}_B_mean", 0) for m in models]
        stds_B = [all_stats[m].get(f"{metric_type}_B_std", 0) for m in models]

        bars1 = ax.bar(x - width / 2, means_A, width, yerr=stds_A,
                       label="Condition A (P-first)", color="steelblue", alpha=0.8,
                       capsize=5)
        bars2 = ax.bar(x + width / 2, means_B, width, yerr=stds_B,
                       label="Condition B (C-first)", color="coral", alpha=0.8,
                       capsize=5)

        ax.set_xlabel("Model")
        ax.set_ylabel("Shannon Entropy (bits)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "comparison_entropy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_comparison_attractors(all_stats: dict, output_dir: str):
    """Side-by-side attractor analysis across models."""
    models = list(all_stats.keys())
    n_models = len(models)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Cross-Model Attractor Analysis (Round Number Bias)", fontsize=14)

    x = np.arange(n_models)
    width = 0.2

    pct10_A = [all_stats[m].get("p_argmax_A_pct_mult10", 0) for m in models]
    pct10_B = [all_stats[m].get("p_argmax_B_pct_mult10", 0) for m in models]
    pct5_A = [all_stats[m].get("p_argmax_A_pct_mult5", 0) for m in models]
    pct5_B = [all_stats[m].get("p_argmax_B_pct_mult5", 0) for m in models]

    ax.bar(x - 1.5 * width, pct10_A, width, label="×10, Cond A", color="steelblue", alpha=0.9)
    ax.bar(x - 0.5 * width, pct10_B, width, label="×10, Cond B", color="steelblue", alpha=0.5)
    ax.bar(x + 0.5 * width, pct5_A, width, label="×5, Cond A", color="coral", alpha=0.9)
    ax.bar(x + 1.5 * width, pct5_B, width, label="×5, Cond B", color="coral", alpha=0.5)

    # Random baseline lines
    ax.axhline(10.9, color="steelblue", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(20.8, color="coral", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(n_models - 0.5, 11.5, "Random ×10 (10.9%)", fontsize=8, color="steelblue")
    ax.text(n_models - 0.5, 21.5, "Random ×5 (20.8%)", fontsize=8, color="coral")

    ax.set_xlabel("Model")
    ax.set_ylabel("% of predictions at round numbers")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "comparison_attractors.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_comparison_order_effects(all_stats: dict, output_dir: str):
    """Order effect comparison (KL divergence) across models."""
    models = list(all_stats.keys())
    n_models = len(models)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Cross-Model Order Effects (KL Divergence A↔B)", fontsize=14)

    x = np.arange(n_models)
    width = 0.35

    p_means = [all_stats[m].get("order_effect_p_mean", 0) for m in models]
    c_means = [all_stats[m].get("order_effect_c_mean", 0) for m in models]
    p_stds = [all_stats[m].get("order_effect_p_std", 0) for m in models]
    c_stds = [all_stats[m].get("order_effect_c_std", 0) for m in models]

    ax.bar(x - width / 2, p_means, width, yerr=p_stds,
           label="P distribution shift", color="steelblue", alpha=0.8, capsize=5)
    ax.bar(x + width / 2, c_means, width, yerr=c_stds,
           label="C distribution shift", color="coral", alpha=0.8, capsize=5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean KL Divergence (nats)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "comparison_order_effects.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_comparison_calibration_error(all_stats: dict, output_dir: str):
    """Calibration error (MAE) comparison across models."""
    models = list(all_stats.keys())
    n_models = len(models)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Cross-Model Calibration Error (MAE, Tier 1 & 2)", fontsize=14)

    x = np.arange(n_models)
    width = 0.35

    mae_A = [all_stats[m].get("p_error_A_mean", 0) for m in models]
    mae_B = [all_stats[m].get("p_error_B_mean", 0) for m in models]
    std_A = [all_stats[m].get("p_error_A_std", 0) for m in models]
    std_B = [all_stats[m].get("p_error_B_std", 0) for m in models]

    ax.bar(x - width / 2, mae_A, width, yerr=std_A,
           label="Condition A (P-first)", color="steelblue", alpha=0.8, capsize=5)
    ax.bar(x + width / 2, mae_B, width, yerr=std_B,
           label="Condition B (C-first)", color="coral", alpha=0.8, capsize=5)

    # Chance baseline
    ax.axhline(33.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(n_models - 0.5, 34, "Chance baseline (≈33)", fontsize=9, color="gray")

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "comparison_calibration_error.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def run_single_model_analysis(df: pd.DataFrame, output_dir: str, model_name: str = ""):
    """Run all per-model analyses and return summary stats."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[+] Generating plots for {model_name or 'model'}...")

    print("  1. Entropy histograms")
    plot_entropy_histograms(df, output_dir, model_name)

    print("  2. Calibration curves")
    plot_calibration_curves(df, output_dir, model_name)

    print("  3. Confidence vs entropy scatter")
    plot_confidence_vs_entropy(df, output_dir, model_name)

    print("  4. Modality analysis")
    plot_modality_analysis(df, output_dir, model_name)

    print("  5. Attractor analysis")
    plot_attractor_analysis(df, output_dir, model_name)

    print("  6. Order effects")
    plot_order_effects(df, output_dir, model_name)

    print("  7. Error by category")
    plot_error_by_tier(df, output_dir, model_name)

    # Summary statistics
    print("\n[+] Computing summary statistics...")
    stats = generate_summary_stats(df, output_dir, model_name)
    print_summary(stats)

    return stats


def run_cross_model_comparison(all_stats: dict, output_dir: str):
    """Generate cross-model comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n[+] Generating cross-model comparison plots...")

    print("  1. Entropy comparison")
    plot_comparison_entropy(all_stats, output_dir)

    print("  2. Attractor comparison")
    plot_comparison_attractors(all_stats, output_dir)

    print("  3. Order effect comparison")
    plot_comparison_order_effects(all_stats, output_dir)

    print("  4. Calibration error comparison")
    plot_comparison_calibration_error(all_stats, output_dir)

    # Save combined stats
    path = Path(output_dir) / "comparison_stats.json"
    with open(path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM Probability Calibration Results")
    parser.add_argument("--results", type=str, default="results/",
                        help="Results directory (single model or parent of model dirs)")
    parser.add_argument("--output", type=str, default="reports/figures/",
                        help="Output directory for figures")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (save only) — default backend is Agg anyway")
    args = parser.parse_args()

    # Find results files
    print("[+] Scanning for results...")
    results_map = find_results_files(args.results)

    if not results_map:
        print(f"[!] No results.jsonl found in {args.results}")
        sys.exit(1)

    print(f"[+] Found {len(results_map)} model(s): {list(results_map.keys())}")

    if len(results_map) == 1 and "default" in results_map:
        # Single model, direct path
        df = load_results(results_map["default"])
        if len(df) == 0:
            print("[!] No results to analyze")
            return
        run_single_model_analysis(df, args.output, model_name="")

    elif len(results_map) == 1:
        # Single model subdirectory
        model_name = list(results_map.keys())[0]
        df = load_results(results_map[model_name])
        if len(df) == 0:
            print("[!] No results to analyze")
            return
        run_single_model_analysis(df, args.output, model_name=model_name)

    else:
        # Multiple models — run per-model + cross-model comparison
        all_stats = {}
        for model_name, results_path in results_map.items():
            df = load_results(results_path)
            if len(df) == 0:
                print(f"[!] Skipping {model_name} — no results")
                continue
            model_output = str(Path(args.output) / model_name)
            stats = run_single_model_analysis(df, model_output, model_name=model_name)
            all_stats[model_name] = stats

        # Cross-model comparison
        if len(all_stats) >= 2:
            comparison_output = str(Path(args.output) / "comparison")
            run_cross_model_comparison(all_stats, comparison_output)

    print(f"\n[+] All figures saved to: {args.output}")


if __name__ == "__main__":
    main()
