from __future__ import annotations

"""Paper-facing analysis for the Independent Cascade approximation benchmarks.

The functions in this module intentionally use only the final benchmark outputs:

    results/real_world_benchmark/benchmark_results.csv
    results/real_world_benchmark/instances.csv
    results/real_world_benchmark/selected_hyperparameters.csv
    results/synthetic_benchmark/benchmark_results.csv
    results/synthetic_benchmark/instances.csv
    results/synthetic_benchmark/selected_hyperparameters.csv

The large calibration-curve CSVs are not required for the paper-level summary
figures and paired comparisons generated here.
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "Naive": "NM",
    "dmp_est": "DMP",
    "dmp_est_r2": "DMP2",
    "dmp_inf": "DMP-inf",
    "swe": "SWE",
    "swe_cavity": "Cavity SWE",
    "swe_no": "SWE-NO",
    "swe_no_cavity": "Cavity SWE-NO",
    "swe_hib_cavity": "SWE-HIB-C",
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

METRICS = ("RMSE", "MAE", "Pearson", "Spearman", "AUC", "runtime_sec")


def paper_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method).replace("_", " "))


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from the current directory or a supplied path."""
    p = Path.cwd() if start is None else Path(start)
    p = p.resolve()
    candidates = [p, *p.parents]
    for candidate in candidates:
        if (
            (candidate / "results" / "real_world_benchmark" / "benchmark_results.csv").exists()
            and (candidate / "results" / "synthetic_benchmark" / "benchmark_results.csv").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root containing results/real_world_benchmark "
        "and results/synthetic_benchmark."
    )


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_one(root: Path, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "results" / benchmark
    raw = pd.read_csv(base / "benchmark_results.csv")
    instances = pd.read_csv(base / "instances.csv")
    selected = pd.read_csv(base / "selected_hyperparameters.csv")

    raw = _coerce_metrics(raw)

    # Keep all rows for audit purposes, then construct the analysis-ready subset.
    ok = raw[raw["status"].astype(str).eq("ok") & raw["RMSE"].notna()].copy()

    metadata_candidates = [
        "instance_id",
        "setting_id",
        "dataset",
        "graph_family",
        "target_mean_degree",
        "realized_mean_degree",
        "edge_model",
        "requested_seed_fraction",
        "realized_seed_fraction",
        "target_secondary_activation",
        "true_secondary_activation",
        "true_mean",
        "edge_prob_mean",
        "edge_prob_std",
        "edge_prob_max",
        "graph_id",
        "rep",
        "graph_rep",
    ]
    join_cols = [c for c in metadata_candidates if c in instances.columns]
    extra_cols = [c for c in join_cols if c not in ok.columns or c == "instance_id"]
    meta = instances[extra_cols].drop_duplicates("instance_id") if extra_cols else instances[["instance_id"]]
    ok = ok.merge(meta, on="instance_id", how="left", validate="many_to_one")

    # Canonical control names used by the analysis functions.
    if "requested_seed_fraction" not in ok.columns:
        ok["requested_seed_fraction"] = pd.to_numeric(
            ok["setting_id"].astype(str).str.extract(r"__s([0-9.]+)", expand=False),
            errors="coerce",
        )
    if "target_secondary_activation" not in ok.columns:
        ok["target_secondary_activation"] = pd.to_numeric(
            ok["setting_id"].astype(str).str.extract(r"__g([0-9.]+)", expand=False),
            errors="coerce",
        )

    ok["benchmark"] = "real" if benchmark == "real_world_benchmark" else "synthetic"
    return ok, raw, selected


def load_all(root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    root = find_repo_root(root)
    real, real_raw, real_selected = _load_one(root, "real_world_benchmark")
    synthetic, synthetic_raw, synthetic_selected = _load_one(root, "synthetic_benchmark")
    return {
        "root": root,
        "real": real,
        "synthetic": synthetic,
        "real_raw": real_raw,
        "synthetic_raw": synthetic_raw,
        "real_selected": real_selected,
        "synthetic_selected": synthetic_selected,
    }


def data_quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Count successful and failed held-out rows for every method."""
    out = (
        raw.groupby(["method", "status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    out.insert(1, "Method", out["method"].map(paper_label))
    out["total"] = out.drop(columns=["method", "Method"]).sum(axis=1, numeric_only=True)
    return out.sort_values("method").reset_index(drop=True)


def pooled_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "N": ("RMSE", "count"),
        "RMSE": ("RMSE", "median"),
        "RMSE_mean": ("RMSE", "mean"),
        "MAE": ("MAE", "median"),
        "Spearman": ("Spearman", "median"),
        "Pearson": ("Pearson", "median"),
        "AUC": ("AUC", "median"),
        "runtime_sec": ("runtime_sec", "median"),
    }
    out = df.groupby("method", as_index=False).agg(**agg)
    out.insert(1, "Method", out["method"].map(paper_label))
    return out.sort_values(["RMSE", "method"], kind="stable").reset_index(drop=True)


def grouped_method_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby([group_col, "method"], as_index=False)
        .agg(
            N=("RMSE", "count"),
            RMSE=("RMSE", "median"),
            MAE=("MAE", "median"),
            Spearman=("Spearman", "median"),
            runtime_sec=("runtime_sec", "median"),
        )
    )
    out.insert(2, "Method", out["method"].map(paper_label))
    return out.sort_values([group_col, "RMSE", "method"], kind="stable").reset_index(drop=True)


def setting_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse test replicates within a setting before pooling across settings."""
    per_setting = (
        df.groupby(["setting_id", "method"], as_index=False)
        .agg(
            RMSE=("RMSE", "median"),
            MAE=("MAE", "median"),
            Spearman=("Spearman", "median"),
            runtime_sec=("runtime_sec", "median"),
        )
    )
    out = (
        per_setting.groupby("method", as_index=False)
        .agg(
            settings=("setting_id", "nunique"),
            RMSE=("RMSE", "median"),
            MAE=("MAE", "median"),
            Spearman=("Spearman", "median"),
            runtime_sec=("runtime_sec", "median"),
        )
    )
    out.insert(1, "Method", out["method"].map(paper_label))
    return out.sort_values(["RMSE", "method"], kind="stable").reset_index(drop=True)


def paired_instances(df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """Pair two methods by held-out instance.

    Positive ``rmse_improvement`` means method_a has lower RMSE than method_b.
    """
    if method_a not in set(df["method"]) or method_b not in set(df["method"]):
        return pd.DataFrame()

    metadata = [
        c for c in (
            "instance_id", "setting_id", "dataset", "graph_family",
            "target_mean_degree", "edge_model", "requested_seed_fraction",
            "target_secondary_activation", "true_secondary_activation",
            "true_mean", "edge_prob_mean", "edge_prob_std", "edge_prob_max",
        ) if c in df.columns
    ]
    key = ["instance_id"]
    left_cols = list(dict.fromkeys(key + metadata + ["RMSE", "runtime_sec", "Spearman", "MAE"]))
    right_cols = key + ["RMSE", "runtime_sec", "Spearman", "MAE"]

    a = df[df["method"].eq(method_a)][left_cols].copy()
    b = df[df["method"].eq(method_b)][right_cols].copy()
    paired = a.merge(b, on=key, how="inner", suffixes=("_a", "_b"), validate="one_to_one")
    if paired.empty:
        return paired

    paired["method_a"] = method_a
    paired["method_b"] = method_b
    paired["label_a"] = paper_label(method_a)
    paired["label_b"] = paper_label(method_b)
    paired["rmse_improvement"] = paired["RMSE_b"] - paired["RMSE_a"]
    paired["rmse_relative_improvement"] = np.where(
        paired["RMSE_b"] > 0,
        paired["rmse_improvement"] / paired["RMSE_b"],
        np.nan,
    )
    paired["a_wins"] = paired["rmse_improvement"] > 0
    paired["tie"] = np.isclose(paired["rmse_improvement"], 0.0, atol=1e-15, rtol=0.0)
    paired["runtime_ratio_a_over_b"] = np.where(
        paired["runtime_sec_b"] > 0,
        paired["runtime_sec_a"] / paired["runtime_sec_b"],
        np.nan,
    )
    return paired


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "method_a": paired["method_a"].iloc[0],
        "Method A": paired["label_a"].iloc[0],
        "method_b": paired["method_b"].iloc[0],
        "Method B": paired["label_b"].iloc[0],
        "N": len(paired),
        "A_win_fraction": float(np.mean(paired["a_wins"])),
        "tie_fraction": float(np.mean(paired["tie"])),
        "median_RMSE_improvement": float(np.median(paired["rmse_improvement"])),
        "mean_RMSE_improvement": float(np.mean(paired["rmse_improvement"])),
        "median_relative_improvement": float(np.nanmedian(paired["rmse_relative_improvement"])),
        "median_runtime_ratio_A_over_B": float(np.nanmedian(paired["runtime_ratio_a_over_b"])),
    }])


def paired_by(paired: pd.DataFrame, group_cols: str | list[str] | tuple[str, ...]) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = [c for c in group_cols if c in paired.columns]
    if not group_cols:
        return pd.DataFrame()
    out = (
        paired.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            N=("rmse_improvement", "size"),
            A_win_fraction=("a_wins", "mean"),
            median_RMSE_improvement=("rmse_improvement", "median"),
            mean_RMSE_improvement=("rmse_improvement", "mean"),
            median_relative_improvement=("rmse_relative_improvement", "median"),
            median_runtime_ratio_A_over_B=("runtime_ratio_a_over_b", "median"),
        )
    )
    return out


def selected_hyperparameter_summary(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected.copy()
    df["param_T"] = pd.to_numeric(df.get("param_T"), errors="coerce")
    df["param_layers"] = pd.to_numeric(df.get("param_layers"), errors="coerce")
    out = (
        df.groupby("method", as_index=False)
        .agg(
            settings=("setting_id", "nunique"),
            median_T=("param_T", "median"),
            min_T=("param_T", "min"),
            max_T=("param_T", "max"),
            median_layers=("param_layers", "median"),
            min_layers=("param_layers", "min"),
            max_layers=("param_layers", "max"),
            median_calibration_RMSE=("mean_RMSE", "median"),
        )
    )
    out.insert(1, "Method", out["method"].map(paper_label))
    return out.sort_values("method").reset_index(drop=True)


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_rmse_boxplot(df: pd.DataFrame, path: Path, title: str) -> None:
    summary = pooled_method_summary(df)
    methods = summary["method"].tolist()
    values = [df.loc[df["method"].eq(m), "RMSE"].dropna().to_numpy() for m in methods]
    labels = [paper_label(m) for m in methods]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.36 * len(methods))))
    ax.boxplot(values, vert=False, labels=labels, showfliers=False, whis=(5, 95))
    ax.set_xlabel("RMSE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    _savefig(fig, path)


def plot_runtime_accuracy(df: pd.DataFrame, path: Path, title: str) -> None:
    s = pooled_method_summary(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s["runtime_sec"], s["RMSE"], s=45)
    for _, row in s.iterrows():
        ax.annotate(row["Method"], (row["runtime_sec"], row["RMSE"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    positive_runtime = s["runtime_sec"].dropna()
    if len(positive_runtime) and np.all(positive_runtime > 0):
        ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_ylabel("Median RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _savefig(fig, path)


def plot_pairwise_control(paired: pd.DataFrame, control: str, path: Path, title: str) -> None:
    if paired.empty or control not in paired.columns:
        return
    s = paired_by(paired, control).sort_values(control)
    if s.empty:
        return
    x = np.arange(len(s))
    labels = [str(v) for v in s[control]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, s["median_RMSE_improvement"], marker="o")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, labels, rotation=30 if len(labels) > 6 else 0)
    ax.set_ylabel("Median RMSE improvement (B - A)")
    ax.set_xlabel(control.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _savefig(fig, path)


def plot_pairwise_heatmap(
    paired: pd.DataFrame,
    row: str,
    col: str,
    path: Path,
    title: str,
) -> None:
    if paired.empty or row not in paired.columns or col not in paired.columns:
        return
    table = paired.groupby([row, col])["rmse_improvement"].median().unstack(col)
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(table.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), [str(x) for x in table.columns])
    ax.set_yticks(np.arange(len(table.index)), [str(x) for x in table.index])
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel(row.replace("_", " "))
    ax.set_title(title)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.4g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Median RMSE improvement (B - A)")
    _savefig(fig, path)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


COMPARISONS = {
    "dmp2_vs_dmp": ("dmp_est_r2", "dmp_est"),
    "ipl_vs_ale": ("modified_ALE", "ALE_heuristic"),
    "ale2_vs_ale": ("ALE2", "ALE_heuristic"),
    "ipl2_vs_ipl": ("modified_ALE2", "modified_ALE"),
    "ipl2_vs_ale2": ("modified_ALE2", "ALE2"),
    "swe_no_vs_swe": ("swe_no", "swe"),
    "swe_no_vs_nm": ("swe_no", "Naive"),
    "cavity_swe_no_vs_swe_no": ("swe_no_cavity", "swe_no"),
    "cavity_swe_vs_swe": ("swe_cavity", "swe"),
    "sss_vs_nm": ("sss", "Naive"),
}


def export_comparison_suite(
    df: pd.DataFrame,
    benchmark_name: str,
    tables_dir: Path,
    figures_dir: Path,
    primary_group: str,
) -> list[dict]:
    claim_rows: list[dict] = []
    for name, (a, b) in COMPARISONS.items():
        paired = paired_instances(df, a, b)
        if paired.empty:
            continue
        base = tables_dir / f"{benchmark_name}__{name}"
        paired_summary(paired).to_csv(base.with_name(base.name + "__overall.csv"), index=False)
        paired_by(paired, primary_group).to_csv(base.with_name(base.name + f"__by_{primary_group}.csv"), index=False)
        paired_by(paired, "edge_model").to_csv(base.with_name(base.name + "__by_edge_model.csv"), index=False)
        paired_by(paired, "requested_seed_fraction").to_csv(base.with_name(base.name + "__by_seed_fraction.csv"), index=False)
        paired_by(paired, "target_secondary_activation").to_csv(base.with_name(base.name + "__by_target_g.csv"), index=False)
        paired_by(paired, ["requested_seed_fraction", "target_secondary_activation"]).to_csv(
            base.with_name(base.name + "__by_seed_and_g.csv"), index=False
        )

        overall = paired_summary(paired).iloc[0].to_dict()
        overall.update({"benchmark": benchmark_name, "comparison": name})
        claim_rows.append(overall)

        # DMP2/DMP gets the full regime figures because it is the principal
        # higher-order comparison in the paper. ALE/IPL and SWE comparisons get
        # one-dimensional regime plots below.
        if name == "dmp2_vs_dmp":
            plot_pairwise_heatmap(
                paired,
                "requested_seed_fraction",
                "target_secondary_activation",
                figures_dir / f"{benchmark_name}__dmp2_vs_dmp__seed_g_heatmap.png",
                f"{benchmark_name.title()}: DMP2 improvement over DMP",
            )
            plot_pairwise_control(
                paired,
                primary_group,
                figures_dir / f"{benchmark_name}__dmp2_vs_dmp__by_{primary_group}.png",
                f"{benchmark_name.title()}: DMP2 improvement over DMP by {primary_group}",
            )

        if name in {"ipl_vs_ale", "ale2_vs_ale", "ipl2_vs_ipl", "swe_no_vs_swe", "cavity_swe_no_vs_swe_no"}:
            plot_pairwise_control(
                paired,
                "target_secondary_activation",
                figures_dir / f"{benchmark_name}__{name}__by_target_g.png",
                f"{benchmark_name.title()}: {paper_label(a)} vs {paper_label(b)} by diffusion strength",
            )
            plot_pairwise_control(
                paired,
                "requested_seed_fraction",
                figures_dir / f"{benchmark_name}__{name}__by_seed_fraction.png",
                f"{benchmark_name.title()}: {paper_label(a)} vs {paper_label(b)} by seed fraction",
            )

    return claim_rows


def generate_all(root: str | Path | None = None) -> dict[str, Path | pd.DataFrame]:
    """Generate the paper-facing tables and figures from final held-out outputs."""
    data = load_all(root)
    root = data["root"]
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    tables_dir = analysis_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Data-quality audit. This is deliberately exported first so stale or failed
    # rows remain visible rather than being silently swallowed by the analysis.
    data_quality_report(data["real_raw"]).to_csv(tables_dir / "real__data_quality.csv", index=False)
    data_quality_report(data["synthetic_raw"]).to_csv(tables_dir / "synthetic__data_quality.csv", index=False)

    claim_rows: list[dict] = []
    for name, df, primary_group in (
        ("real", data["real"], "dataset"),
        ("synthetic", data["synthetic"], "graph_family"),
    ):
        pooled_method_summary(df).to_csv(tables_dir / f"{name}__pooled_method_summary.csv", index=False)
        setting_level_summary(df).to_csv(tables_dir / f"{name}__setting_level_summary.csv", index=False)
        grouped_method_summary(df, primary_group).to_csv(tables_dir / f"{name}__by_{primary_group}.csv", index=False)
        grouped_method_summary(df, "edge_model").to_csv(tables_dir / f"{name}__by_edge_model.csv", index=False)
        grouped_method_summary(df, "requested_seed_fraction").to_csv(tables_dir / f"{name}__by_seed_fraction.csv", index=False)
        grouped_method_summary(df, "target_secondary_activation").to_csv(tables_dir / f"{name}__by_target_g.csv", index=False)

        plot_pooled_rmse_boxplot(
            df,
            figures_dir / f"{name}__pooled_rmse_boxplot.png",
            f"{name.title()} benchmark: held-out RMSE",
        )
        plot_runtime_accuracy(
            df,
            figures_dir / f"{name}__runtime_accuracy.png",
            f"{name.title()} benchmark: runtime-accuracy trade-off",
        )

        claim_rows.extend(
            export_comparison_suite(df, name, tables_dir, figures_dir, primary_group)
        )

    selected_hyperparameter_summary(data["real_selected"]).to_csv(
        tables_dir / "real__selected_hyperparameters.csv", index=False
    )
    selected_hyperparameter_summary(data["synthetic_selected"]).to_csv(
        tables_dir / "synthetic__selected_hyperparameters.csv", index=False
    )

    claims = pd.DataFrame(claim_rows)
    claims.to_csv(tables_dir / "paired_comparison_summary.csv", index=False)

    # A concise machine-readable availability table is useful because historical
    # benchmark archives may not contain every newer method (notably ALE2/IPL2
    # in older synthetic runs).
    availability_rows = []
    for name, df in (("real", data["real"]), ("synthetic", data["synthetic"])):
        methods = set(df["method"].unique())
        for internal, label in METHOD_LABELS.items():
            availability_rows.append({
                "benchmark": name,
                "method": internal,
                "Method": label,
                "available": internal in methods,
            })
    pd.DataFrame(availability_rows).to_csv(tables_dir / "method_availability.csv", index=False)

    return {
        "root": root,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "real_summary": pooled_method_summary(data["real"]),
        "synthetic_summary": pooled_method_summary(data["synthetic"]),
        "paired_summary": claims,
    }
