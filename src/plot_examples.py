#!/usr/bin/env python3
"""
Generate per-question distribution example plots for LLM probability calibration.

Selects diverse examples across tiers showing different distribution properties:
- Well-calibrated (p_argmax close to GT)
- Badly wrong (large p_error)
- Low entropy (peaked)
- High entropy (diffuse)
- Multimodal (many modes)
- Strong order effect (high KL divergence between conditions)

For each question: 2×2 subplot (P/C × Condition A/B).
Also generates a summary gallery figure.

Usage:
    python src/plot_examples.py \
      --results /path/to/results.jsonl \
      --output /path/to/output/dir/ \
      --n 9 --no-show
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from JSONL into DataFrame."""
    records = []
    with open(results_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                for key in list(entry.keys()):
                    if key.endswith("_dist_A") or key.endswith("_dist_B"):
                        entry[key] = np.array(entry[key])
                records.append(entry)
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    print(f"[+] Loaded {len(df)} results from {results_path}")
    return df


def select_diverse_examples(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """Select diverse examples covering different distribution properties."""
    selected_indices = []
    used_categories = set()

    # 1. Well-calibrated: smallest p_error_A
    tier12 = df[df["tier"].isin([1, 2])].copy()
    if len(tier12) > 0:
        best_calibrated = tier12.loc[tier12["p_error_A"].idxmin()]
        selected_indices.append(best_calibrated.name)
        used_categories.add(best_calibrated["category"])

    # 2. Badly wrong: largest p_error_A
    if len(tier12) > 0:
        worst = tier12.loc[tier12["p_error_A"].idxmax()]
        if worst.name not in selected_indices:
            selected_indices.append(worst.name)
            used_categories.add(worst["category"])

    # 3. Very low entropy (peaked) — smallest p_entropy_A
    lowest_ent = df.loc[df["p_entropy_A"].idxmin()]
    if lowest_ent.name not in selected_indices:
        selected_indices.append(lowest_ent.name)
        used_categories.add(lowest_ent["category"])

    # 4. High entropy (diffuse) — largest p_entropy_A
    highest_ent = df.loc[df["p_entropy_A"].idxmax()]
    if highest_ent.name not in selected_indices:
        selected_indices.append(highest_ent.name)
        used_categories.add(highest_ent["category"])

    # 5. Strong multimodality — most modes in p distribution
    most_modes = df.loc[df["n_modes_p_A"].idxmax()]
    if most_modes.name not in selected_indices:
        selected_indices.append(most_modes.name)
        used_categories.add(most_modes["category"])

    # 6. Strong order effect — largest order_effect_p
    strongest_order = df.loc[df["order_effect_p"].idxmax()]
    if strongest_order.name not in selected_indices:
        selected_indices.append(strongest_order.name)
        used_categories.add(strongest_order["category"])

    # Ensure at least one from each tier
    for tier in [1, 2, 3]:
        tier_df = df[df["tier"] == tier]
        if len(tier_df) == 0:
            continue
        tier_in_selection = [i for i in selected_indices if df.loc[i, "tier"] == tier]
        if not tier_in_selection:
            # Pick an interesting one from this tier (highest entropy not yet selected)
            candidates = tier_df[~tier_df.index.isin(selected_indices)]
            if len(candidates) > 0:
                pick = candidates.loc[candidates["p_entropy_A"].idxmax()]
                selected_indices.append(pick.name)
                used_categories.add(pick["category"])

    # Fill remaining slots to reach n with diverse categories
    remaining = df[~df.index.isin(selected_indices)]
    while len(selected_indices) < n and len(remaining) > 0:
        # Prefer categories not yet covered
        uncovered = remaining[~remaining["category"].isin(used_categories)]
        if len(uncovered) > 0:
            # Pick the one with highest entropy (interesting distributions)
            pick = uncovered.loc[uncovered["p_entropy_A"].idxmax()]
        else:
            # Pick highest order effect from remaining
            pick = remaining.loc[remaining["order_effect_p"].idxmax()]
        selected_indices.append(pick.name)
        used_categories.add(pick["category"])
        remaining = df[~df.index.isin(selected_indices)]

    return df.loc[selected_indices].reset_index(drop=True)


def get_example_label(row: pd.Series) -> str:
    """Generate a descriptive label for why this example was selected."""
    labels = []

    # Check calibration
    if row.get("p_error_A") is not None:
        if row["p_error_A"] <= 2:
            labels.append("well-calibrated")
        elif row["p_error_A"] >= 40:
            labels.append("badly miscalibrated")

    # Check entropy
    if row["p_entropy_A"] < 0.5:
        labels.append("very peaked (low entropy)")
    elif row["p_entropy_A"] > 3.0:
        labels.append("highly diffuse (high entropy)")

    # Check multimodality
    if row["n_modes_p_A"] >= 4:
        labels.append(f"multimodal ({row['n_modes_p_A']} modes)")

    # Check order effect
    if row["order_effect_p"] > 1.0:
        labels.append("strong order effect")

    if not labels:
        labels.append(f"tier {row['tier']}, {row['category']}")

    return "; ".join(labels)


def plot_single_example(row: pd.Series, idx: int, output_dir: str) -> str:
    """Generate a 2×2 figure for a single question. Returns filename."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Wrap question text
    q_text = textwrap.fill(row["question"], width=90)
    gt = row.get("ground_truth", None)

    # Metadata line
    meta_parts = []
    if gt is not None:
        meta_parts.append(f"GT={gt:.1f}")
    meta_parts.append(f"p_argmax_A={row['p_argmax_A']}")
    meta_parts.append(f"c_argmax_A={row['c_argmax_A']}")
    meta_parts.append(f"H(P_A)={row['p_entropy_A']:.2f}")
    meta_parts.append(f"modes_P_A={row['n_modes_p_A']}")
    meta_parts.append(f"order_KL={row['order_effect_p']:.4f}")
    meta_line = " | ".join(meta_parts)

    fig.suptitle(f"{q_text}\n{meta_line}", fontsize=10, wrap=True)

    # Plot configurations: (row, col, dist_key, argmax_key, entropy_key, modes_key, title)
    configs = [
        (0, 0, "p_dist_A", "p_argmax_A", "p_entropy_A", "n_modes_p_A", "P dist (Cond A: P-first)"),
        (0, 1, "c_dist_A", "c_argmax_A", "c_entropy_A", "n_modes_c_A", "C dist (Cond A: P-first)"),
        (1, 0, "p_dist_B", "p_argmax_B", "p_entropy_B", "n_modes_p_B", "P dist (Cond B: C-first)"),
        (1, 1, "c_dist_B", "c_argmax_B", "c_entropy_B", "n_modes_c_B", "C dist (Cond B: C-first)"),
    ]

    x = np.arange(101)

    for r, c, dist_key, argmax_key, ent_key, modes_key, title in configs:
        ax = axes[r, c]
        dist = row[dist_key]

        if dist is None or (isinstance(dist, np.ndarray) and len(dist) == 0):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Bar chart
        colors = ["steelblue"] * 101
        argmax_val = int(row[argmax_key])
        if 0 <= argmax_val <= 100:
            colors[argmax_val] = "red"

        ax.bar(x, dist, color=colors, width=1.0, alpha=0.8)

        # Ground truth line
        if gt is not None:
            ax.axvline(gt, color="green", linestyle="--", linewidth=2, alpha=0.8,
                       label=f"GT={gt:.1f}")

        # Argmax marker
        ax.axvline(argmax_val, color="red", linestyle="-", linewidth=1.5, alpha=0.6,
                   label=f"argmax={argmax_val}")

        # Subplot label
        entropy_val = row[ent_key]
        modes_val = row[modes_key]
        label_text = f"argmax={argmax_val} | H={entropy_val:.2f} | modes={modes_val}"
        ax.set_title(f"{title}\n{label_text}", fontsize=9)

        ax.set_xlabel("Value (0–100)")
        ax.set_ylabel("Probability")
        ax.set_xlim(-1, 101)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    # Generate filename
    category = row["category"].replace(" ", "_").replace("/", "_")
    tier = int(row["tier"])
    filename = f"example_q{idx}_tier{tier}_{category}.png"
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")
    return filename


def plot_gallery(selected: pd.DataFrame, output_dir: str) -> str:
    """Generate a gallery figure showing P-dist (Cond A) for all selected examples."""
    n = len(selected)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    fig.suptitle("Gallery: P-Distribution (Condition A) — Selected Examples", fontsize=14)

    if nrows == 1:
        axes = axes.reshape(1, -1)

    x = np.arange(101)

    for idx, (_, row) in enumerate(selected.iterrows()):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        dist = row["p_dist_A"]
        if dist is None or (isinstance(dist, np.ndarray) and len(dist) == 0):
            ax.set_visible(False)
            continue

        # Bar chart
        ax.bar(x, dist, color="steelblue", width=1.0, alpha=0.7)

        # GT line
        gt = row.get("ground_truth", None)
        if gt is not None:
            ax.axvline(gt, color="green", linestyle="--", linewidth=2, alpha=0.8)

        # Argmax
        argmax_val = int(row["p_argmax_A"])
        ax.axvline(argmax_val, color="red", linestyle="-", linewidth=1.5, alpha=0.6)

        # Title: abbreviated question + key info
        q_short = row["question"][:60] + ("..." if len(row["question"]) > 60 else "")
        category = row["category"]
        tier = int(row["tier"])
        ax.set_title(f"T{tier}/{category}\n{q_short}", fontsize=8)
        ax.set_xlim(-1, 101)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    filename = "gallery_p_dist_cond_A.png"
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Plot per-question distribution examples")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for figures")
    parser.add_argument("--n", type=int, default=9,
                        help="Number of examples to select (default: 9)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (save only)")
    args = parser.parse_args()

    # Load data
    df = load_results(args.results)
    if len(df) == 0:
        print("[!] No results to plot")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Select diverse examples
    print(f"\n[+] Selecting {args.n} diverse examples...")
    selected = select_diverse_examples(df, n=args.n)
    print(f"  Selected {len(selected)} examples:")
    for i, (_, row) in enumerate(selected.iterrows()):
        label = get_example_label(row)
        print(f"    [{i}] Tier {row['tier']} / {row['category']}: {label}")
        print(f"        Q: {row['question'][:80]}...")

    # Generate per-question figures
    print(f"\n[+] Generating per-question figures...")
    filenames = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        fname = plot_single_example(row, idx, args.output)
        filenames.append(fname)

    # Generate gallery figure
    print(f"\n[+] Generating gallery figure...")
    gallery_fname = plot_gallery(selected, args.output)

    # Save selection metadata
    metadata = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        meta = {
            "index": idx,
            "id": row.get("id", ""),
            "tier": int(row["tier"]),
            "category": row["category"],
            "question": row["question"],
            "ground_truth": float(row["ground_truth"]) if row.get("ground_truth") is not None else None,
            "p_argmax_A": int(row["p_argmax_A"]),
            "c_argmax_A": int(row["c_argmax_A"]),
            "p_argmax_B": int(row["p_argmax_B"]),
            "c_argmax_B": int(row["c_argmax_B"]),
            "p_entropy_A": float(row["p_entropy_A"]),
            "n_modes_p_A": int(row["n_modes_p_A"]),
            "order_effect_p": float(row["order_effect_p"]),
            "p_error_A": float(row["p_error_A"]) if row.get("p_error_A") is not None else None,
            "label": get_example_label(row),
            "filename": filenames[idx],
        }
        metadata.append(meta)

    meta_path = Path(args.output) / "selection_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[+] Saved selection metadata: {meta_path}")

    print(f"\n[+] Done! {len(filenames)} example figures + 1 gallery saved to {args.output}")


if __name__ == "__main__":
    main()
