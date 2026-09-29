#!/usr/bin/env python3
"""
Create paper-ready summary figures and pooled-median tables from the
held-out IC benchmark CSV files.

Expected inputs
---------------
Real-world benchmark CSV:
    results/real_world_benchmark/benchmark_results.csv

Synthetic benchmark CSV:
    results/synthetic_benchmark/benchmark_results.csv

Typical use
-----------
python make_paper_benchmark_outputs.py \
    --real results/real_world_benchmark/benchmark_results.csv \
    --synthetic results/synthetic_benchmark/benchmark_results.csv \
    --out paper_benchmark_outputs

Optional diffusion-regime breakdowns:
python make_paper_benchmark_outputs.py \
    --real results/real_world_benchmark/benchmark_results.csv \
    --synthetic results/synthetic_benchmark/benchmark_results.csv \
    --out paper_benchmark_outputs \
    --by-g

Outputs
-------
For each benchmark:
    <name>_pooled_rmse.pdf
    <name>_pooled_rmse.png
    <name>_pooled_metric_medians.csv
    <name>_pooled_metric_medians.tex

If --by-g is supplied, one additional RMSE boxplot is produced for each
target secondary-activation regime.

Notes
-----
* All pooled summaries operate on held-out benchmark rows with status == "ok".
* Methods are ordered by pooled median RMSE.
* Boxplot whiskers are the 5th and 95th percentiles.
* Outlier points are suppressed in the main figure to keep it readable.
* Exact RMSE == 0 rows are retained in all numerical summaries. They cannot
  be displayed at x=0 on a logarithmic axis, so they are written to an audit
  CSV if present.
* The table pools instance-level rows directly. It does NOT first average
  within datasets, so datasets/settings with more feasible held-out rows
  contribute more observations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paper-facing method names
# ---------------------------------------------------------------------

METHOD_LABELS = {
    "Naive": "NM",
    "dmp_est": "DMP",
    "dmp_est_r2": r"DMP$_2$",
    "dmp_inf": r"DMP$_{\infty}$",
    "swe": "SWE",
    "swe_cavity": "Cavity SWE",
    "swe_no": "SWE-NO",
    "swe_no_cavity": "Cavity SWE-NO",
    "swe_hib_cavity": "SWE-HIB cavity",
    "additive_swe": "SWE-add",
    "additive_swe_cavity": "Cavity SWE-add",
    "ALE_heuristic": "ALE",
    "ALE2": "ALE2",
    "modified_ALE": "IPL",
    "modified_ALE2": "IPL2",
    "SPM": "SPM",
    "SP1M": "SP1M",
    "sss": "SSS",
    "sss_noself": "SSS-Noself",
    "mia": "MIA",
}


METRICS = ["RMSE", "MAE", "Spearman", "Pearson", "AUC", "runtime_sec"]


# ---------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------

def load_benchmark(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required = {
        "setting_id",
        "instance_id",
        "method",
        "RMSE",
        "MAE",
        "Pearson",
        "Spearman",
        "AUC",
        "runtime_sec",
        "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    # The benchmark writer records failed method evaluations as separate rows.
    # They are not part of the held-out performance distribution.
    df = df[df["status"].astype(str).eq("ok")].copy()

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows for which RMSE exists. Other metrics may legitimately be NaN.
    df = df[df["RMSE"].notna()].copy()

    # Parse experimental controls from setting_id for optional breakdowns.
    df["target_g"] = pd.to_numeric(
        df["setting_id"].str.extract(r"__g([0-9.]+)", expand=False),
        errors="coerce",
    )
    df["seed_fraction"] = pd.to_numeric(
        df["setting_id"].str.extract(r"__s([0-9.]+)", expand=False),
        errors="coerce",
    )

    return df


def paper_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


# ---------------------------------------------------------------------
# Pooled summary table
# ---------------------------------------------------------------------

def pooled_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per method. N counts successful held-out evaluations.
    Each metric is pooled directly across those held-out rows.
    """
    g = df.groupby("method", sort=False)

    out = g.agg(
        N=("RMSE", "count"),
        RMSE=("RMSE", "median"),
        MAE=("MAE", "median"),
        Spearman=("Spearman", "median"),
        Pearson=("Pearson", "median"),
        AUC=("AUC", "median"),
        runtime_sec=("runtime_sec", "median"),
    ).reset_index()

    out = out.sort_values(
        ["RMSE", "method"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    out.insert(
        1,
        "Method",
        out["method"].map(paper_label),
    )
    return out


def write_summary_table(
    summary: pd.DataFrame,
    stem: Path,
    benchmark_title: str,
) -> None:
    # Raw machine-readable values.
    summary.to_csv(
        stem.with_suffix(".csv"),
        index=False,
    )

    # Paper-facing formatted copy.
    table = summary[
        [
            "Method",
            "N",
            "RMSE",
            "MAE",
            "Spearman",
            "Pearson",
            "AUC",
            "runtime_sec",
        ]
    ].copy()

    table = table.rename(
        columns={"runtime_sec": "Runtime (s)"}
    )

    # String formatting is intentional here so the LaTeX output has
    # consistent precision and remains easy to paste into the paper.
    table["N"] = table["N"].astype(int).astype(str)
    table["RMSE"] = table["RMSE"].map(lambda x: f"{x:.4f}")
    table["MAE"] = table["MAE"].map(lambda x: f"{x:.4f}")
    table["Spearman"] = table["Spearman"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.5f}"
    )
    table["Pearson"] = table["Pearson"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.5f}"
    )
    table["AUC"] = table["AUC"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.6f}"
    )
    table["Runtime (s)"] = table["Runtime (s)"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.4g}"
    )

    caption = (
        f"Pooled held-out performance on the {benchmark_title} benchmark. "
        "Entries are medians over all successful held-out instance-level "
        "evaluations. Lower is better for RMSE, MAE, and runtime; higher is "
        "better for Spearman, Pearson, and AUC."
    )

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrr",
        caption=caption,
        label=f"tab:{stem.stem.replace('_pooled_metric_medians', '')}-pooled-metrics",
        position="t",
    )

    stem.with_suffix(".tex").write_text(
        latex,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# RMSE figure
# ---------------------------------------------------------------------

def rmse_boxplot(
    df: pd.DataFrame,
    out_stem: Path,
    title: str,
    *,
    show_reference_lines: bool = True,
) -> None:
    med = (
        df.groupby("method")["RMSE"]
        .median()
        .sort_values(ascending=False)
    )

    methods = med.index.tolist()
    data = [
        df.loc[df["method"].eq(method), "RMSE"]
        .dropna()
        .to_numpy(dtype=float)
        for method in methods
    ]
    labels = [paper_label(m) for m in methods]

    # Scale height with number of methods but keep reasonable bounds.
    fig_height = max(5.0, 0.45 * len(methods) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        widths=0.58,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Held-out RMSE")
    ax.set_ylabel("Method")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    # Absolute-error guides. These are directly interpretable because
    # activation probabilities lie in [0,1].
    if show_reference_lines:
        for x, label in [
            (0.01, "RMSE = 0.01"),
            (0.05, "RMSE = 0.05"),
        ]:
            ax.axvline(x, linestyle="--", linewidth=0.9, alpha=0.55)

    fig.tight_layout()

    fig.savefig(
        Path(str(out_stem) + ".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        Path(str(out_stem) + ".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Optional breakdown by target secondary activation g
# ---------------------------------------------------------------------

def make_g_breakdowns(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    title_prefix: str,
) -> None:
    values = sorted(
        float(x)
        for x in df["target_g"].dropna().unique()
    )

    for g in values:
        part = df[np.isclose(df["target_g"], g)].copy()
        if part.empty:
            continue

        rmse_boxplot(
            part,
            out_dir / f"{prefix}_rmse_g{int(round(100*g)):02d}",
            f"{title_prefix}: held-out RMSE, $g^*={g:.2f}$",
        )


# ---------------------------------------------------------------------
# One benchmark
# ---------------------------------------------------------------------

def process_one(
    csv_path: str | Path,
    out_dir: Path,
    prefix: str,
    title: str,
    *,
    by_g: bool,
) -> None:
    df = load_benchmark(csv_path)

    # Audit rows that cannot appear at x=0 on a log axis.
    zero_rows = df[df["RMSE"].eq(0.0)].copy()
    if not zero_rows.empty:
        audit_path = out_dir / f"{prefix}_zero_rmse_rows.csv"
        zero_rows.to_csv(audit_path, index=False)
        print(
            f"[{prefix}] WARNING: found {len(zero_rows)} exact RMSE=0 rows. "
            f"They are retained in tables but x=0 is not visible on a log axis. "
            f"Audit written to {audit_path}"
        )

    summary = pooled_medians(df)

    write_summary_table(
        summary,
        out_dir / f"{prefix}_pooled_metric_medians",
        title,
    )

    rmse_boxplot(
        df,
        out_dir / f"{prefix}_pooled_rmse",
        f"{title}: held-out RMSE",
    )

    if by_g:
        make_g_breakdowns(
            df,
            out_dir,
            prefix,
            title,
        )

    print(
        f"[{prefix}] {len(df):,} successful held-out rows, "
        f"{df['method'].nunique()} methods."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real",
        type=Path,
        required=True,
        help="Real-world benchmark_results.csv",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        required=True,
        help="Synthetic benchmark_results.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper_benchmark_outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--by-g",
        action="store_true",
        help=(
            "Also produce separate RMSE boxplots for "
            "g*=0.10, 0.35, and 0.60."
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    process_one(
        args.real,
        args.out,
        prefix="real_world",
        title="Real-world datasets",
        by_g=args.by_g,
    )

    process_one(
        args.synthetic,
        args.out,
        prefix="synthetic",
        title="Synthetic datasets",
        by_g=args.by_g,
    )


if __name__ == "__main__":
    main()
