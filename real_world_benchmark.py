
"""
Final real-world benchmark for IC approximation methods.

Expected protocol
-----------------
The real graph topology is fixed for each dataset.  Independent benchmark
instances are therefore created by varying

    * the seed set;
    * the stochastic Dirichlet edge-weight realization.

For constant and weighted-cascade edge models, only the seed set changes.

For each setting

    (dataset, edge_model, seed_fraction, target_secondary_activation)

we create K_cal calibration instances and K_test held-out test instances.

For each method:
    1. Run the full candidate parameter grid on calibration instances.
    2. Choose ONE parameter combination using mean calibration RMSE.
    3. Freeze it.
    4. Evaluate it unchanged on held-out instances.

The full calibration curves are retained.

The benchmark helper utilities and tuning logic are defined locally so this
module does not depend on the exploratory scripts or on the synthetic benchmark.

Typical default MC budgets
--------------------------
Calibration:
    NetHept     10k
    WikiVote    10k
    CondMat      8k
    citHepTh     8k
    Enron        6k
    Epinions     5k
    Slashdot     5k

Held-out test:
    NetHept     30k
    WikiVote    30k
    CondMat     20k
    citHepTh    20k
    Enron       15k
    Epinions    10k
    Slashdot    10k

These are deliberately asymmetric.  If a small number of final comparisons
are extremely close, rerun those cells at 50k+ rather than paying that price
for every setting.
"""

from __future__ import annotations

import copy
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd




# =============================================================================
# Self-contained benchmark utilities
# =============================================================================

def derive_seed(base_seed: int, *indices: int) -> int:
    """
    Deterministically derive an independent uint32 seed from integer indices.

    Unlike repeatedly calling np.random.default_rng(base_seed), this produces
    genuinely different but reproducible random streams for graph/edge/seed/MC
    components.
    """
    ss = np.random.SeedSequence([int(base_seed), *map(int, indices)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def json_dumps(x) -> str:
    return json.dumps(
        x,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_or_blank(x):
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if np.isfinite(v) else ""


def probability_array(values, n: int) -> np.ndarray:
    """
    Convert array-like or node->probability dict into an n-vector.

    Assumes ICGraph node ids are 0,...,n-1 when a dict is supplied.
    """
    if isinstance(values, dict):
        return np.asarray([values[i] for i in range(n)], dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"Expected probability vector of shape ({n},), got {arr.shape}.")
    return arr


def secondary_activation(final_probs, prior_probs) -> float:
    """
    g = (mean(final)-mean(prior)) / (1-mean(prior)).
    """
    final = np.asarray(final_probs, dtype=np.float64)
    prior = np.asarray(prior_probs, dtype=np.float64)

    p0 = float(np.mean(prior))
    pf = float(np.mean(final))

    if p0 >= 1.0:
        return 0.0

    return float((pf - p0) / (1.0 - p0))


def _summary_stats(prefix: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_q99": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_zero_fraction": 0.0,
        }

    mean = float(np.mean(x))
    q50, q90, q99 = np.quantile(x, [0.50, 0.90, 0.99])

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_cv": float(np.std(x) / mean) if mean > 0.0 else 0.0,
        f"{prefix}_q50": float(q50),
        f"{prefix}_q90": float(q90),
        f"{prefix}_q99": float(q99),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_zero_fraction": float(np.mean(x == 0.0)),
    }


def basic_graph_characteristics(graph) -> dict:
    """
    Cheap O(n+m) graph features that are safe to compute even for large graphs.

    Clustering, triangles, components, and modularity are deliberately not
    computed here because the preferred implementation depends on the graph
    stack. Use the benchmark's extra_graph_features_fn hook when needed.
    """
    n = int(graph.n)
    m = int(graph.m)

    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)

    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)

    try:
        rev = np.asarray(graph.ensure_reverse(), dtype=np.int64)
        reciprocal_edge_fraction = float(np.mean(rev >= 0)) if m else 0.0
    except Exception:
        reciprocal_edge_fraction = float("nan")

    memory_gb = ""
    if hasattr(graph, "memory_gb"):
        try:
            memory_gb = float(graph.memory_gb())
        except Exception:
            pass

    density = (
        float(m / (n * (n - 1)))
        if n > 1
        else 0.0
    )

    out = {
        "n": n,
        "m": m,
        "m_per_n": float(m / n) if n else 0.0,
        "directed_arc_density": density,
        "reciprocal_edge_fraction": reciprocal_edge_fraction,
        "graph_memory_gb": memory_gb,
    }

    out.update(_summary_stats("in_degree", indeg))
    out.update(_summary_stats("out_degree", outdeg))

    if n:
        out["in_out_degree_corr"] = (
            float(np.corrcoef(indeg, outdeg)[0, 1])
            if np.std(indeg) > 0 and np.std(outdeg) > 0
            else 0.0
        )
    else:
        out["in_out_degree_corr"] = 0.0

    return out


def make_exact_seed_prior(
    n: int,
    seed_fraction: float,
    rng_seed: int,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select exactly round(seed_fraction*n) deterministic seed nodes.

    This is preferable to Bernoulli selection when '1%', '5%', ... are meant to
    be controlled experimental conditions.
    """
    n = int(n)
    frac = float(seed_fraction)

    if not (0.0 <= frac <= 1.0):
        raise ValueError("seed_fraction must lie in [0,1].")

    k = int(round(frac * n))
    if frac > 0.0 and k == 0:
        k = 1

    rng = np.random.default_rng(int(rng_seed))
    selected = (
        np.sort(rng.choice(n, size=k, replace=False))
        if k > 0
        else np.empty(0, dtype=np.int64)
    )

    prior = np.zeros(n, dtype=dtype)
    prior[selected] = 1.0

    return prior, selected


def seed_topology_characteristics(graph, selected: np.ndarray) -> dict:
    n = int(graph.n)
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)

    indeg = np.bincount(dst, minlength=n).astype(np.float64)
    outdeg = np.bincount(src, minlength=n).astype(np.float64)

    selected = np.asarray(selected, dtype=np.int64)

    if selected.size == 0:
        return {
            "seed_mean_in_degree": 0.0,
            "seed_mean_out_degree": 0.0,
            "seed_in_degree_ratio_to_graph": 0.0,
            "seed_out_degree_ratio_to_graph": 0.0,
        }

    mean_in = float(np.mean(indeg))
    mean_out = float(np.mean(outdeg))
    seed_in = float(np.mean(indeg[selected]))
    seed_out = float(np.mean(outdeg[selected]))

    return {
        "seed_mean_in_degree": seed_in,
        "seed_mean_out_degree": seed_out,
        "seed_in_degree_ratio_to_graph": (
            seed_in / mean_in if mean_in > 0 else 0.0
        ),
        "seed_out_degree_ratio_to_graph": (
            seed_out / mean_out if mean_out > 0 else 0.0
        ),
    }


def probability_vector_summary(prefix: str, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    q10, q50, q90, q99 = np.quantile(p, [0.10, 0.50, 0.90, 0.99])

    return {
        f"{prefix}_mean": float(np.mean(p)),
        f"{prefix}_std": float(np.std(p)),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q90": float(q90),
        f"{prefix}_q99": float(q99),
        f"{prefix}_min": float(np.min(p)),
        f"{prefix}_max": float(np.max(p)),
    }


class CsvAppender:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: dict):
        clean = {key: row.get(key, "") for key in self.fieldnames}

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)
            f.flush()


def _canonical_metric_key(k: str) -> str:
    s = str(k).strip().lower().replace("_", "").replace("-", "")
    mapping = {
        "rmse": "RMSE",
        "mae": "MAE",
        "pearson": "Pearson",
        "pearsonr": "Pearson",
        "spearman": "Spearman",
        "spearmanr": "Spearman",
        "auc": "AUC",
        "rocauc": "AUC",
    }
    return mapping.get(s, str(k))


def normalize_metrics(metrics) -> dict:
    if metrics is None:
        return {}

    if not isinstance(metrics, dict):
        raise TypeError(
            "evaluate_predictions() should return a dict for this logger."
        )

    return {
        _canonical_metric_key(k): _finite_or_blank(v)
        for k, v in metrics.items()
    }


def make_pilot_evaluator(
    graph,
    prior_probs,
    optimized_independent_cascade,
):
    """
    Return pilot_eval(edge_prob, num_sim, seed) without mutating the base graph.
    """
    def pilot_eval(edge_prob, num_sim, seed):
        graph_local = copy.copy(graph)
        graph_local.prob = np.asarray(
            edge_prob,
            dtype=np.asarray(graph.prob).dtype,
        )

        return optimized_independent_cascade(
            graph_local,
            prior_probs,
            int(num_sim),
            seed=int(seed),
        )

    return pilot_eval


# =============================================================================
# Benchmark method grids and shared benchmark helpers
# =============================================================================

DEFAULT_METHODS = (
    "swe_no",
    "swe_no_cavity",
    "additive_swe",
    "additive_swe_cavity",
    "swe",
    "swe_cavity",
    "swe_hib_cavity",
    "Naive",
    "dmp_est_r2",
    "dmp_est",
    "dmp_inf",
    "modified_ALE",
    "ALE_heuristic",
    "cavity_ALE",
    "modified_ALE_cavity",
    "SPM",
    "SP1M",
    "sss",
)


TUNED_T_METHODS = {
    "additive_swe",
    "additive_swe_cavity",
    "swe",
    "swe_cavity",
    "swe_hib_cavity",
    "Naive",
    "dmp_est_r2",
    "dmp_est",
    "modified_ALE",
    "ALE_heuristic",
    "cavity_ALE",
    "modified_ALE_cavity",
}


COUPLED_T_LAYER_METHODS = {
    "swe_no",
    "swe_no_cavity",
}


FIXED_METHODS = {
    "dmp_inf",
    "SPM",
    "SP1M",
    "sss",
}


def build_default_parameter_grids(
    *,
    methods=DEFAULT_METHODS,
    T_values=range(1, 21),
    layer_values=range(0, 6),
):
    """
    Generic parameter grid.

    If one of your methods uses a different convention, edit this dictionary in
    the notebook before running the benchmark.
    """
    T_values = tuple(int(x) for x in T_values)
    layer_values = tuple(int(x) for x in layer_values)

    grids = {}

    for method in methods:
        if method in COUPLED_T_LAYER_METHODS:
            grids[method] = [
                {"T": T, "layers": L}
                for T in T_values
                for L in layer_values
            ]
        elif method in TUNED_T_METHODS:
            grids[method] = [
                {"T": T}
                for T in T_values
            ]
        else:
            grids[method] = [{}]

    return grids


def _params_json(params):
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
    )


def _param_fields(params):
    return {
        "params_json": _params_json(params),
        "param_T": params.get("T", ""),
        "param_layers": params.get(
            "layers",
            "",
        ),
    }


def _safe_metric_dict(
    evaluate_predictions,
    predictions,
    true_probs,
):
    raw = evaluate_predictions(
        predictions,
        true_probs,
    )

    metrics = normalize_metrics(
        raw
    )

    return {
        "RMSE": metrics.get(
            "RMSE",
            np.nan,
        ),
        "MAE": metrics.get(
            "MAE",
            np.nan,
        ),
        "Pearson": metrics.get(
            "Pearson",
            np.nan,
        ),
        "Spearman": metrics.get(
            "Spearman",
            np.nan,
        ),
        "AUC": metrics.get(
            "AUC",
            np.nan,
        ),
    }


def _run_fixed(
    run_method_fixed,
    method,
    graph,
    prior_probs,
    params,
):
    """
    Normalize either:
        predictions
    or:
        (predictions, runtime_sec)
    """
    if method == "dmp_est_r2":
        graph.ensure_r2_triangles()
        
    start = time.perf_counter()

    out = run_method_fixed(
        method,
        graph,
        prior_probs,
        dict(params),
    )

    measured_runtime = (
        time.perf_counter() - start
    )

    if (
        isinstance(out, tuple)
        and len(out) == 2
        and np.isscalar(out[1])
    ):
        predictions = out[0]
        runtime = float(out[1])
    else:
        predictions = out
        runtime = measured_runtime

    return predictions, runtime


def _existing_key_set(
    path,
    columns,
):
    path = Path(path)

    if not path.exists():
        return set()

    try:
        df = pd.read_csv(
            path,
            usecols=columns,
        )
    except Exception:
        return set()

    return set(
        tuple(row)
        for row in df[
            columns
        ].itertuples(
            index=False,
            name=None,
        )
    )


def prepare_diffusion_instance(
    *,
    graph,
    graph_meta,
    graph_id,
    split,
    graph_rep,
    edge_model,
    edge_seed,
    seed_fraction,
    seedset_seed,
    target_g,
    pilot_seed,
    make_base_probabilities,
    calibrate_secondary_activation_grid,
    optimized_independent_cascade,
    edge_probability_summary,
    num_sim,
    mc_seed,
    dirichlet_concentration=0.5,
    pilot_num_sim=250,
    pilot_repeats=1,
    binary_search_steps=10,
    target_tol=0.02,
):
    """
    Build one calibrated diffusion instance and compute final MC truth.

    Edge-strength calibration uses only the small pilot MC.
    """
    prior_probs, selected = (
        make_exact_seed_prior(
            graph.n,
            seed_fraction,
            seedset_seed,
            dtype=np.float32,
        )
    )

    seed_features = (
        seed_topology_characteristics(
            graph,
            selected,
        )
    )

    base_prob = (
        make_base_probabilities(
            graph,
            edge_model=edge_model,
            heterogeneous_seed=edge_seed,
            dirichlet_concentration=(
                dirichlet_concentration
            ),
        )
    )

    pilot_eval = (
        make_pilot_evaluator(
            graph,
            prior_probs,
            optimized_independent_cascade,
        )
    )

    calibrated = (
        calibrate_secondary_activation_grid(
            graph,
            base_prob,
            prior_probs,
            targets=(float(target_g),),
            pilot_evaluator=pilot_eval,
            pilot_num_sim=pilot_num_sim,
            pilot_repeats=pilot_repeats,
            pilot_seed=pilot_seed,
            binary_search_steps=binary_search_steps,
            target_tol=target_tol,
        )
    )

    entry = next(
        iter(
            calibrated.values()
        )
    )

    edge_probs = np.asarray(
        entry["prob"],
        dtype=np.asarray(
            graph.prob
        ).dtype,
    )

    info = dict(
        entry["info"]
    )

    instance_graph = copy.copy(
        graph
    )
    instance_graph.prob = (
        edge_probs
    )

    start = time.perf_counter()

    true_raw = (
        optimized_independent_cascade(
            instance_graph,
            prior_probs,
            int(num_sim),
            seed=mc_seed,
        )
    )

    mc_runtime = (
        time.perf_counter()
        - start
    )

    true_probs = (
        probability_array(
            true_raw,
            graph.n,
        )
    )

    true_summary = (
        probability_vector_summary(
            "true",
            true_probs,
        )
    )

    edge_summary = (
        edge_probability_summary(
            edge_probs
        )
    )

    return {
        "graph": instance_graph,
        "prior_probs": prior_probs,
        "true_probs": true_probs,
        "selected": selected,
        "seed_features": seed_features,
        "edge_summary": edge_summary,
        "calibration_info": info,
        "true_summary": true_summary,
        "true_secondary_activation": (
            secondary_activation(
                true_probs,
                prior_probs,
            )
        ),
        "mc_runtime_sec": mc_runtime,
    }


# =============================================================================
# Default MC budgets
# =============================================================================

DEFAULT_CALIBRATION_MC = {
    "NetHept": 10_000,
    "WikiVote": 10_000,
    "Enron": 6_000,
    "Epinions": 5_000,
    "Slashdot": 5_000,
    "CondMat": 8_000,
    "citHepTh": 8_000,
}

DEFAULT_TEST_MC = {
    "NetHept": 30_000,
    "WikiVote": 30_000,
    "Enron": 15_000,
    "Epinions": 10_000,
    "Slashdot": 10_000,
    "CondMat": 20_000,
    "citHepTh": 20_000,
}


# =============================================================================
# CSV schemas
# =============================================================================

GRAPH_FIELDS = [
    "graph_id",
    "dataset",
    "load_seed",
    "n",
    "m",
    "m_per_n",
    "directed_arc_density",
    "reciprocal_edge_fraction",
    "graph_memory_gb",
    "graph_preprocessing_time_sec",
    "graph_features_json",
]

INSTANCE_FIELDS = [
    "instance_id",
    "setting_id",
    "graph_id",
    "dataset",
    "split",
    "rep",
    "edge_model",
    "edge_seed",
    "dirichlet_concentration",
    "requested_seed_fraction",
    "realized_seed_fraction",
    "seed_count",
    "seedset_seed",
    "target_secondary_activation",
    "pilot_seed",
    "alpha",
    "pilot_achieved_secondary_activation",
    "pilot_secondary_activation_abs_error",
    "calibration_clipped_fraction",
    "edge_prob_mean",
    "edge_prob_std",
    "edge_prob_q50",
    "edge_prob_q90",
    "edge_prob_q99",
    "edge_prob_max",
    "edge_prob_frac_ge_0_1",
    "edge_prob_frac_ge_0_25",
    "edge_prob_frac_ge_0_5",
    "true_mean",
    "true_std",
    "true_q10",
    "true_q50",
    "true_q90",
    "true_secondary_activation",
    "mc_num_sim",
    "mc_seed",
    "mc_runtime_sec",
    "seed_features_json",
    "calibration_info_json",
    "status",
    "error_type",
    "error_message",
]

CALIBRATION_FIELDS = [
    "setting_id",
    "instance_id",
    "dataset",
    "method",
    "params_json",
    "param_T",
    "param_layers",
    "RMSE",
    "MAE",
    "Pearson",
    "Spearman",
    "AUC",
    "runtime_sec",
    "status",
    "error_type",
    "error_message",
]

SELECTED_FIELDS = [
    "setting_id",
    "dataset",
    "method",
    "params_json",
    "param_T",
    "param_layers",
    "mean_RMSE",
    "median_RMSE",
    "std_RMSE",
    "mean_runtime_sec",
    "n_calibration_instances",
    "selection_tolerance",
    "best_mean_RMSE",
]

BENCHMARK_FIELDS = [
    "setting_id",
    "instance_id",
    "dataset",
    "method",
    "params_json",
    "param_T",
    "param_layers",
    "RMSE",
    "MAE",
    "Pearson",
    "Spearman",
    "AUC",
    "runtime_sec",
    "status",
    "error_type",
    "error_message",
]


# =============================================================================
# Helpers
# =============================================================================

def _resolve_mc_budget(spec, dataset, graph):
    """
    spec may be:
        int
        dict keyed by dataset, optionally with "_default"
        callable(dataset, graph) -> int
    """
    if callable(spec):
        return int(spec(dataset, graph))

    if isinstance(spec, dict):
        if dataset in spec:
            return int(spec[dataset])
        if "_default" in spec:
            return int(spec["_default"])
        raise KeyError(
            f"No MC budget supplied for dataset {dataset!r}."
        )

    return int(spec)


def _memory_gb(graph):
    if hasattr(graph, "memory_gb"):
        try:
            return float(graph.memory_gb())
        except Exception:
            pass

    total = 0
    for name in ("src", "dst", "prob"):
        arr = getattr(graph, name, None)
        if isinstance(arr, np.ndarray):
            total += arr.nbytes

    return total / (1024 ** 3)


def _select_complete_params(
    records,
    *,
    parameter_grid,
    min_calibration_instances,
    selection_tolerance=0.0,
):
    """
    Select by mean calibration RMSE, but only among parameter combinations
    successfully evaluated on at least min_calibration_instances.

    selection_tolerance is additive:
        mean_RMSE <= best_mean_RMSE + selection_tolerance

    Within the eligible near-optimal set, prefer smaller T, then fewer layers.
    With tolerance=0 this is the usual calibration argmin with deterministic
    complexity-aware tie breaking.
    """
    if len(records) == 0:
        return None, None

    df = pd.DataFrame(records)

    if df.empty:
        return None, None

    ok = (
        df["status"].eq("ok")
        & pd.to_numeric(df["RMSE"], errors="coerce").notna()
    )

    df = df[ok].copy()

    if df.empty:
        return None, None

    grouped = (
        df.groupby("params_json", as_index=False)
        .agg(
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            std_RMSE=("RMSE", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            n_calibration_instances=("instance_id", "nunique"),
        )
    )

    grouped = grouped[
        grouped["n_calibration_instances"]
        >= int(min_calibration_instances)
    ].copy()

    if grouped.empty:
        return None, None

    best = float(grouped["mean_RMSE"].min())

    eligible = grouped[
        grouped["mean_RMSE"]
        <= best + float(selection_tolerance) + 1e-15
    ].copy()

    params_lookup = {
        _params_json(params): dict(params)
        for params in parameter_grid
    }

    def key_for(idx):
        row = eligible.loc[idx]
        params = params_lookup[row["params_json"]]

        T = params.get("T", -1)
        L = params.get("layers", -1)

        return (
            float(T) if T != "" else -1.0,
            float(L) if L != "" else -1.0,
            float(row["mean_RMSE"]),
        )

    chosen_idx = min(eligible.index, key=key_for)
    chosen = eligible.loc[chosen_idx]

    params = params_lookup[chosen["params_json"]]

    selection_row = {
        **_param_fields(params),
        "mean_RMSE": float(chosen["mean_RMSE"]),
        "median_RMSE": float(chosen["median_RMSE"]),
        "std_RMSE": (
            float(chosen["std_RMSE"])
            if pd.notna(chosen["std_RMSE"])
            else np.nan
        ),
        "mean_runtime_sec": float(chosen["mean_runtime_sec"]),
        "n_calibration_instances": int(
            chosen["n_calibration_instances"]
        ),
        "selection_tolerance": float(selection_tolerance),
        "best_mean_RMSE": best,
    }

    return params, selection_row


def _instance_row(
    *,
    instance_id,
    setting_id,
    graph_id,
    dataset,
    split,
    rep,
    edge_model,
    edge_seed,
    dirichlet_concentration,
    seed_fraction,
    seedset_seed,
    target_g,
    pilot_seed,
    mc_num_sim,
    mc_seed,
    inst=None,
    error=None,
):
    base = {
        "instance_id": instance_id,
        "setting_id": setting_id,
        "graph_id": graph_id,
        "dataset": dataset,
        "split": split,
        "rep": rep,
        "edge_model": edge_model,
        "edge_seed": edge_seed,
        "dirichlet_concentration": dirichlet_concentration,
        "requested_seed_fraction": seed_fraction,
        "seedset_seed": seedset_seed,
        "target_secondary_activation": target_g,
        "pilot_seed": pilot_seed,
        "mc_num_sim": mc_num_sim,
        "mc_seed": mc_seed,
    }

    if error is not None:
        return {
            **base,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error)[:1000],
        }

    info = inst["calibration_info"]
    es = inst["edge_summary"]
    ts = inst["true_summary"]

    return {
        **base,
        "realized_seed_fraction": (
            len(inst["selected"]) / inst["graph"].n
        ),
        "seed_count": len(inst["selected"]),
        "alpha": info.get("alpha", ""),
        "pilot_achieved_secondary_activation": info.get(
            "achieved_secondary_activation", ""
        ),
        "pilot_secondary_activation_abs_error": info.get(
            "secondary_activation_abs_error", ""
        ),
        "calibration_clipped_fraction": info.get(
            "clipped_fraction", ""
        ),
        "edge_prob_mean": es.get(
            "edge_prob_mean",
            es.get("mean", ""),
        ),
        "edge_prob_std": es.get(
            "edge_prob_std",
            es.get("std", ""),
        ),
        "edge_prob_q50": es.get(
            "edge_prob_q50",
            es.get("q50", ""),
        ),
        "edge_prob_q90": es.get(
            "edge_prob_q90",
            es.get("q90", ""),
        ),
        "edge_prob_q99": es.get(
            "edge_prob_q99",
            es.get("q99", ""),
        ),
        "edge_prob_max": es.get(
            "edge_prob_max",
            es.get("max", ""),
        ),
        "edge_prob_frac_ge_0_1": es.get(
            "edge_prob_frac_ge_0_1", ""
        ),
        "edge_prob_frac_ge_0_25": es.get(
            "edge_prob_frac_ge_0_25", ""
        ),
        "edge_prob_frac_ge_0_5": es.get(
            "edge_prob_frac_ge_0_5", ""
        ),
        "true_mean": ts.get("true_mean", ""),
        "true_std": ts.get("true_std", ""),
        "true_q10": ts.get("true_q10", ""),
        "true_q50": ts.get("true_q50", ""),
        "true_q90": ts.get("true_q90", ""),
        "true_secondary_activation": inst[
            "true_secondary_activation"
        ],
        "mc_runtime_sec": inst["mc_runtime_sec"],
        "seed_features_json": json_dumps(
            inst["seed_features"]
        ),
        "calibration_info_json": json_dumps(info),
        "status": "ok",
        "error_type": "",
        "error_message": "",
    }


# =============================================================================
# Runner
# =============================================================================

def run_real_world_benchmark(
    *,
    load_dataset,
    make_base_probabilities,
    calibrate_secondary_activation_grid,
    optimized_independent_cascade,
    run_method_fixed,
    evaluate_predictions,
    edge_probability_summary,
    output_dir="results/real_world_benchmark",
    methods=DEFAULT_METHODS,
    parameter_grids=None,
    datasets=(
        "NetHept",
        "WikiVote",
        "Enron",
        "Epinions",
        "Slashdot",
        "CondMat",
        "citHepTh",
    ),
    edge_models=(
        "constant",
        "weighted_cascade",
        "heterogeneous_dirichlet",
    ),
    dirichlet_concentration=0.5,
    seed_fractions=(
        0.01,
        0.05,
        0.10,
        0.20,
    ),
    target_secondary_activations=(
        0.10,
        0.35,
        0.60,
    ),
    num_calibration_instances=3,
    num_test_instances=5,
    calibration_mc=DEFAULT_CALIBRATION_MC,
    test_mc=DEFAULT_TEST_MC,
    pilot_num_sim=250,
    pilot_repeats=1,
    binary_search_steps=10,
    target_tol=0.02,
    selection_tolerance=0.0,
    min_calibration_instances=None,
    base_seed=42,
    extra_graph_features_fn=None,
    graph_precompute_fn=None,
    resume=True,
):
    """
    Run the final held-out real-world benchmark.

    Unlike the synthetic benchmark, topology is fixed within each dataset.
    Calibration/test independence comes from fresh seed sets and, for the
    Dirichlet condition, fresh heterogeneous edge-weight realizations.

    Constant and weighted-cascade edge shapes are deterministic conditional on
    the graph, so only seed sets vary for those conditions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parameter_grids is None:
        parameter_grids = build_default_parameter_grids(
            methods=methods,
            T_values=range(1, 21),
            layer_values=range(0, 6),
        )

    if min_calibration_instances is None:
        min_calibration_instances = int(
            num_calibration_instances
        )

    graphs_path = output_dir / "graphs.csv"
    instances_path = output_dir / "instances.csv"
    calibration_path = output_dir / "calibration_curves.csv"
    selected_path = output_dir / "selected_hyperparameters.csv"
    benchmark_path = output_dir / "benchmark_results.csv"

    graph_writer = CsvAppender(
        graphs_path,
        GRAPH_FIELDS,
    )
    instance_writer = CsvAppender(
        instances_path,
        INSTANCE_FIELDS,
    )
    calibration_writer = CsvAppender(
        calibration_path,
        CALIBRATION_FIELDS,
    )
    selected_writer = CsvAppender(
        selected_path,
        SELECTED_FIELDS,
    )
    benchmark_writer = CsvAppender(
        benchmark_path,
        BENCHMARK_FIELDS,
    )

    graph_done = (
        _existing_key_set(
            graphs_path,
            ["graph_id"],
        )
        if resume
        else set()
    )

    instance_done = (
        _existing_key_set(
            instances_path,
            ["instance_id"],
        )
        if resume
        else set()
    )

    calibration_done = (
        _existing_key_set(
            calibration_path,
            ["instance_id", "method", "params_json"],
        )
        if resume
        else set()
    )

    selected_done = (
        _existing_key_set(
            selected_path,
            ["setting_id", "method"],
        )
        if resume
        else set()
    )

    benchmark_done = (
        _existing_key_set(
            benchmark_path,
            ["instance_id", "method"],
        )
        if resume
        else set()
    )

    dataset_to_idx = {
        d: i
        for i, d in enumerate(datasets)
    }

    # =========================================================================
    # Dataset loop
    # =========================================================================
    for dataset in datasets:
        dataset_idx = dataset_to_idx[dataset]

        print(
            f"\n{'=' * 88}\n"
            f"REAL-WORLD BENCHMARK: {dataset}\n"
            f"{'=' * 88}"
        )

        load_seed = derive_seed(
            base_seed,
            10,
            dataset_idx,
        )

        graph = load_dataset(
            dataset,
            seed=load_seed,
        )

        graph_id = f"realbench__{dataset}"

        graph_precompute_time = 0.0

        if graph_precompute_fn is not None:
            start = time.perf_counter()

            graph_precompute_fn(
                graph,
                dataset,
            )

            graph_precompute_time = (
                time.perf_counter() - start
            )

        graph_features = basic_graph_characteristics(
            graph
        )

        extra_features = {}

        if extra_graph_features_fn is not None:
            extra_features = dict(
                extra_graph_features_fn(
                    graph,
                    dataset,
                )
            )

        if (graph_id,) not in graph_done:
            graph_writer.append(
                {
                    "graph_id": graph_id,
                    "dataset": dataset,
                    "load_seed": load_seed,
                    "n": graph.n,
                    "m": graph.m,
                    "m_per_n": (
                        float(graph.m)
                        / float(graph.n)
                    ),
                    "directed_arc_density": graph_features.get(
                        "directed_arc_density", ""
                    ),
                    "reciprocal_edge_fraction": graph_features.get(
                        "reciprocal_edge_fraction", ""
                    ),
                    "graph_memory_gb": graph_features.get(
                        "graph_memory_gb",
                        _memory_gb(graph),
                    ),
                    "graph_preprocessing_time_sec": (
                        graph_precompute_time
                    ),
                    "graph_features_json": json_dumps(
                        {
                            **graph_features,
                            **extra_features,
                        }
                    ),
                }
            )

            graph_done.add((graph_id,))

        calibration_num_sim = _resolve_mc_budget(
            calibration_mc,
            dataset,
            graph,
        )

        test_num_sim = _resolve_mc_budget(
            test_mc,
            dataset,
            graph,
        )

        print(
            f"n={graph.n:,}, m={graph.m:,}, "
            f"m/n={graph.m / graph.n:.3f}"
        )
        print(
            f"MC budget: calibration={calibration_num_sim:,}, "
            f"test={test_num_sim:,}"
        )

        # =====================================================================
        # CALIBRATION INSTANCES
        # =====================================================================
        for rep in range(
            int(num_calibration_instances)
        ):
            split_code = 0
            split = "calibration"

            print(
                f"\n--- {dataset}: calibration replicate "
                f"{rep + 1}/{num_calibration_instances} ---"
            )

            # Same seed set is paired across edge models within this replicate.
            seedset_seeds = {}

            for sf_idx, sf in enumerate(seed_fractions):
                seedset_seeds[float(sf)] = derive_seed(
                    base_seed,
                    20,
                    dataset_idx,
                    split_code,
                    rep,
                    sf_idx,
                )

            # Stochastic edge shape changes across replicate and edge model, but
            # is reused across seed fractions and g values.
            for edge_idx, edge_model in enumerate(
                edge_models
            ):
                edge_seed = derive_seed(
                    base_seed,
                    30,
                    dataset_idx,
                    split_code,
                    rep,
                    edge_idx,
                )

                for sf_idx, sf in enumerate(
                    seed_fractions
                ):
                    seedset_seed = seedset_seeds[
                        float(sf)
                    ]

                    for target_idx, target_g in enumerate(
                        target_secondary_activations
                    ):
                        target_g = float(target_g)

                        setting_id = (
                            f"{dataset}"
                            f"__{edge_model}"
                            f"__s{float(sf):.3f}"
                            f"__g{target_g:.2f}"
                        )

                        instance_id = (
                            f"{setting_id}"
                            f"__cal"
                            f"__r{rep:02d}"
                        )

                        # If all calibration parameter rows already exist,
                        # no need to rebuild MC truth for this instance.
                        expected_keys = [
                            (
                                instance_id,
                                method,
                                _params_json(params),
                            )
                            for method in methods
                            for params in parameter_grids[method]
                        ]

                        if (
                            expected_keys
                            and all(
                                key in calibration_done
                                for key in expected_keys
                            )
                        ):
                            continue

                        pilot_seed = derive_seed(
                            base_seed,
                            40,
                            dataset_idx,
                            split_code,
                            rep,
                            edge_idx,
                            sf_idx,
                            target_idx,
                        )

                        mc_seed = derive_seed(
                            base_seed,
                            50,
                            dataset_idx,
                            split_code,
                            rep,
                            edge_idx,
                            sf_idx,
                            target_idx,
                        )

                        print(
                            f"[CAL] {edge_model}, "
                            f"s={float(sf):.2f}, g={target_g:.2f}"
                        )

                        try:
                            inst = prepare_diffusion_instance(
                                graph=graph,
                                graph_meta={"dataset": dataset},
                                graph_id=graph_id,
                                split=split,
                                graph_rep=rep,
                                edge_model=edge_model,
                                edge_seed=edge_seed,
                                seed_fraction=sf,
                                seedset_seed=seedset_seed,
                                target_g=target_g,
                                pilot_seed=pilot_seed,
                                make_base_probabilities=(
                                    make_base_probabilities
                                ),
                                calibrate_secondary_activation_grid=(
                                    calibrate_secondary_activation_grid
                                ),
                                optimized_independent_cascade=(
                                    optimized_independent_cascade
                                ),
                                edge_probability_summary=(
                                    edge_probability_summary
                                ),
                                num_sim=calibration_num_sim,
                                mc_seed=mc_seed,
                                dirichlet_concentration=(
                                    dirichlet_concentration
                                ),
                                pilot_num_sim=pilot_num_sim,
                                pilot_repeats=pilot_repeats,
                                binary_search_steps=(
                                    binary_search_steps
                                ),
                                target_tol=target_tol,
                            )

                        except Exception as exc:
                            print(
                                f"  [INSTANCE ERROR] "
                                f"{type(exc).__name__}: {exc}"
                            )

                            if (
                                (instance_id,)
                                not in instance_done
                            ):
                                instance_writer.append(
                                    _instance_row(
                                        instance_id=instance_id,
                                        setting_id=setting_id,
                                        graph_id=graph_id,
                                        dataset=dataset,
                                        split=split,
                                        rep=rep,
                                        edge_model=edge_model,
                                        edge_seed=edge_seed,
                                        dirichlet_concentration=(
                                            dirichlet_concentration
                                        ),
                                        seed_fraction=sf,
                                        seedset_seed=seedset_seed,
                                        target_g=target_g,
                                        pilot_seed=pilot_seed,
                                        mc_num_sim=(
                                            calibration_num_sim
                                        ),
                                        mc_seed=mc_seed,
                                        error=exc,
                                    )
                                )

                                instance_done.add(
                                    (instance_id,)
                                )

                            continue

                        if (
                            (instance_id,)
                            not in instance_done
                        ):
                            instance_writer.append(
                                _instance_row(
                                    instance_id=instance_id,
                                    setting_id=setting_id,
                                    graph_id=graph_id,
                                    dataset=dataset,
                                    split=split,
                                    rep=rep,
                                    edge_model=edge_model,
                                    edge_seed=edge_seed,
                                    dirichlet_concentration=(
                                        dirichlet_concentration
                                    ),
                                    seed_fraction=sf,
                                    seedset_seed=seedset_seed,
                                    target_g=target_g,
                                    pilot_seed=pilot_seed,
                                    mc_num_sim=(
                                        calibration_num_sim
                                    ),
                                    mc_seed=mc_seed,
                                    inst=inst,
                                )
                            )

                            instance_done.add(
                                (instance_id,)
                            )

                        # -----------------------------------------------------
                        # Full parameter curve for each method.
                        # -----------------------------------------------------
                        for method in methods:
                            for params in parameter_grids[
                                method
                            ]:
                                pjson = _params_json(
                                    params
                                )

                                done_key = (
                                    instance_id,
                                    method,
                                    pjson,
                                )

                                if (
                                    done_key
                                    in calibration_done
                                ):
                                    continue

                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter(
                                            "ignore"
                                        )

                                        preds, runtime = _run_fixed(
                                            run_method_fixed,
                                            method,
                                            inst["graph"],
                                            inst["prior_probs"],
                                            params,
                                        )

                                    preds = np.asarray(
                                        preds
                                        if not isinstance(preds, dict)
                                        else [
                                            preds[i]
                                            for i in range(
                                                graph.n
                                            )
                                        ],
                                        dtype=np.float64,
                                    )

                                    metrics = _safe_metric_dict(
                                        evaluate_predictions,
                                        preds,
                                        inst["true_probs"],
                                    )

                                    row = {
                                        "setting_id": setting_id,
                                        "instance_id": instance_id,
                                        "dataset": dataset,
                                        "method": method,
                                        **_param_fields(params),
                                        **metrics,
                                        "runtime_sec": runtime,
                                        "status": "ok",
                                        "error_type": "",
                                        "error_message": "",
                                    }

                                except Exception as exc:
                                    print(
                                        f"  [METHOD ERROR] "
                                        f"{method} {params}: "
                                        f"{type(exc).__name__}: {exc}"
                                    )

                                    row = {
                                        "setting_id": setting_id,
                                        "instance_id": instance_id,
                                        "dataset": dataset,
                                        "method": method,
                                        **_param_fields(params),
                                        "RMSE": np.nan,
                                        "MAE": np.nan,
                                        "Pearson": np.nan,
                                        "Spearman": np.nan,
                                        "AUC": np.nan,
                                        "runtime_sec": np.nan,
                                        "status": "error",
                                        "error_type": (
                                            type(exc).__name__
                                        ),
                                        "error_message": (
                                            str(exc)[:1000]
                                        ),
                                    }

                                calibration_writer.append(
                                    row
                                )

                                calibration_done.add(
                                    done_key
                                )

        # =====================================================================
        # SELECT FROZEN HYPERPARAMETERS
        # =====================================================================
        if calibration_path.exists():
            cal_df = pd.read_csv(
                calibration_path
            )

            cal_df = cal_df[
                cal_df["dataset"].astype(str).eq(
                    dataset
                )
            ].copy()
        else:
            cal_df = pd.DataFrame()

        selected_params = {}

        for edge_model in edge_models:
            for sf in seed_fractions:
                for target_g in target_secondary_activations:
                    target_g = float(target_g)

                    setting_id = (
                        f"{dataset}"
                        f"__{edge_model}"
                        f"__s{float(sf):.3f}"
                        f"__g{target_g:.2f}"
                    )

                    for method in methods:
                        if cal_df.empty:
                            rows = []
                        else:
                            rows = (
                                cal_df[
                                    (
                                        cal_df["setting_id"]
                                        == setting_id
                                    )
                                    & (
                                        cal_df["method"]
                                        == method
                                    )
                                ]
                                .to_dict("records")
                            )

                        params, selection = (
                            _select_complete_params(
                                rows,
                                parameter_grid=(
                                    parameter_grids[
                                        method
                                    ]
                                ),
                                min_calibration_instances=(
                                    min_calibration_instances
                                ),
                                selection_tolerance=(
                                    selection_tolerance
                                ),
                            )
                        )

                        if params is None:
                            # This naturally catches topologically infeasible
                            # targets such as WikiVote g=.35/.60.
                            continue

                        key = (
                            setting_id,
                            method,
                        )

                        selected_params[
                            key
                        ] = params

                        if key not in selected_done:
                            selected_writer.append(
                                {
                                    "setting_id": setting_id,
                                    "dataset": dataset,
                                    "method": method,
                                    **selection,
                                }
                            )

                            selected_done.add(
                                key
                            )

        n_selected_settings = len(
            {
                setting
                for setting, _method
                in selected_params.keys()
            }
        )

        print(
            f"\nSelected parameters for "
            f"{len(selected_params):,} setting×method pairs "
            f"across {n_selected_settings:,} feasible settings."
        )

        # =====================================================================
        # HELD-OUT TEST INSTANCES
        # =====================================================================
        for rep in range(
            int(num_test_instances)
        ):
            split_code = 1
            split = "test"

            print(
                f"\n--- {dataset}: held-out replicate "
                f"{rep + 1}/{num_test_instances} ---"
            )

            seedset_seeds = {}

            for sf_idx, sf in enumerate(seed_fractions):
                seedset_seeds[float(sf)] = derive_seed(
                    base_seed,
                    20,
                    dataset_idx,
                    split_code,
                    rep,
                    sf_idx,
                )

            for edge_idx, edge_model in enumerate(
                edge_models
            ):
                edge_seed = derive_seed(
                    base_seed,
                    30,
                    dataset_idx,
                    split_code,
                    rep,
                    edge_idx,
                )

                for sf_idx, sf in enumerate(
                    seed_fractions
                ):
                    seedset_seed = seedset_seeds[
                        float(sf)
                    ]

                    for target_idx, target_g in enumerate(
                        target_secondary_activations
                    ):
                        target_g = float(target_g)

                        setting_id = (
                            f"{dataset}"
                            f"__{edge_model}"
                            f"__s{float(sf):.3f}"
                            f"__g{target_g:.2f}"
                        )

                        applicable_methods = [
                            method
                            for method in methods
                            if (
                                setting_id,
                                method,
                            ) in selected_params
                        ]

                        # Infeasible / incomplete calibration setting.
                        if not applicable_methods:
                            continue

                        instance_id = (
                            f"{setting_id}"
                            f"__test"
                            f"__r{rep:02d}"
                        )

                        expected_keys = [
                            (
                                instance_id,
                                method,
                            )
                            for method in applicable_methods
                        ]

                        if (
                            expected_keys
                            and all(
                                key in benchmark_done
                                for key in expected_keys
                            )
                        ):
                            continue

                        pilot_seed = derive_seed(
                            base_seed,
                            40,
                            dataset_idx,
                            split_code,
                            rep,
                            edge_idx,
                            sf_idx,
                            target_idx,
                        )

                        mc_seed = derive_seed(
                            base_seed,
                            50,
                            dataset_idx,
                            split_code,
                            rep,
                            edge_idx,
                            sf_idx,
                            target_idx,
                        )

                        print(
                            f"[TEST] {edge_model}, "
                            f"s={float(sf):.2f}, g={target_g:.2f}"
                        )

                        try:
                            inst = prepare_diffusion_instance(
                                graph=graph,
                                graph_meta={"dataset": dataset},
                                graph_id=graph_id,
                                split=split,
                                graph_rep=rep,
                                edge_model=edge_model,
                                edge_seed=edge_seed,
                                seed_fraction=sf,
                                seedset_seed=seedset_seed,
                                target_g=target_g,
                                pilot_seed=pilot_seed,
                                make_base_probabilities=(
                                    make_base_probabilities
                                ),
                                calibrate_secondary_activation_grid=(
                                    calibrate_secondary_activation_grid
                                ),
                                optimized_independent_cascade=(
                                    optimized_independent_cascade
                                ),
                                edge_probability_summary=(
                                    edge_probability_summary
                                ),
                                num_sim=test_num_sim,
                                mc_seed=mc_seed,
                                dirichlet_concentration=(
                                    dirichlet_concentration
                                ),
                                pilot_num_sim=pilot_num_sim,
                                pilot_repeats=pilot_repeats,
                                binary_search_steps=(
                                    binary_search_steps
                                ),
                                target_tol=target_tol,
                            )

                        except Exception as exc:
                            print(
                                f"  [INSTANCE ERROR] "
                                f"{type(exc).__name__}: {exc}"
                            )

                            if (
                                (instance_id,)
                                not in instance_done
                            ):
                                instance_writer.append(
                                    _instance_row(
                                        instance_id=instance_id,
                                        setting_id=setting_id,
                                        graph_id=graph_id,
                                        dataset=dataset,
                                        split=split,
                                        rep=rep,
                                        edge_model=edge_model,
                                        edge_seed=edge_seed,
                                        dirichlet_concentration=(
                                            dirichlet_concentration
                                        ),
                                        seed_fraction=sf,
                                        seedset_seed=seedset_seed,
                                        target_g=target_g,
                                        pilot_seed=pilot_seed,
                                        mc_num_sim=(
                                            test_num_sim
                                        ),
                                        mc_seed=mc_seed,
                                        error=exc,
                                    )
                                )

                                instance_done.add(
                                    (instance_id,)
                                )

                            continue

                        if (
                            (instance_id,)
                            not in instance_done
                        ):
                            instance_writer.append(
                                _instance_row(
                                    instance_id=instance_id,
                                    setting_id=setting_id,
                                    graph_id=graph_id,
                                    dataset=dataset,
                                    split=split,
                                    rep=rep,
                                    edge_model=edge_model,
                                    edge_seed=edge_seed,
                                    dirichlet_concentration=(
                                        dirichlet_concentration
                                    ),
                                    seed_fraction=sf,
                                    seedset_seed=seedset_seed,
                                    target_g=target_g,
                                    pilot_seed=pilot_seed,
                                    mc_num_sim=test_num_sim,
                                    mc_seed=mc_seed,
                                    inst=inst,
                                )
                            )

                            instance_done.add(
                                (instance_id,)
                            )

                        # -----------------------------------------------------
                        # Frozen held-out evaluation.
                        # -----------------------------------------------------
                        for method in applicable_methods:
                            done_key = (
                                instance_id,
                                method,
                            )

                            if done_key in benchmark_done:
                                continue

                            params = selected_params[
                                (
                                    setting_id,
                                    method,
                                )
                            ]

                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter(
                                        "ignore"
                                    )

                                    preds, runtime = _run_fixed(
                                        run_method_fixed,
                                        method,
                                        inst["graph"],
                                        inst["prior_probs"],
                                        params,
                                    )

                                preds = np.asarray(
                                    preds
                                    if not isinstance(preds, dict)
                                    else [
                                        preds[i]
                                        for i in range(
                                            graph.n
                                        )
                                    ],
                                    dtype=np.float64,
                                )

                                metrics = _safe_metric_dict(
                                    evaluate_predictions,
                                    preds,
                                    inst["true_probs"],
                                )

                                row = {
                                    "setting_id": setting_id,
                                    "instance_id": instance_id,
                                    "dataset": dataset,
                                    "method": method,
                                    **_param_fields(params),
                                    **metrics,
                                    "runtime_sec": runtime,
                                    "status": "ok",
                                    "error_type": "",
                                    "error_message": "",
                                }

                            except Exception as exc:
                                print(
                                    f"  [METHOD ERROR] "
                                    f"{method} {params}: "
                                    f"{type(exc).__name__}: {exc}"
                                )

                                row = {
                                    "setting_id": setting_id,
                                    "instance_id": instance_id,
                                    "dataset": dataset,
                                    "method": method,
                                    **_param_fields(params),
                                    "RMSE": np.nan,
                                    "MAE": np.nan,
                                    "Pearson": np.nan,
                                    "Spearman": np.nan,
                                    "AUC": np.nan,
                                    "runtime_sec": np.nan,
                                    "status": "error",
                                    "error_type": (
                                        type(exc).__name__
                                    ),
                                    "error_message": (
                                        str(exc)[:1000]
                                    ),
                                }

                            benchmark_writer.append(
                                row
                            )

                            benchmark_done.add(
                                done_key
                            )

        print(
            f"\nCompleted dataset {dataset}."
        )

    print(
        "\nReal-world benchmark finished.\n"
        f"  graphs:        {graphs_path}\n"
        f"  instances:     {instances_path}\n"
        f"  tuning curves: {calibration_path}\n"
        f"  selected:      {selected_path}\n"
        f"  benchmark:     {benchmark_path}"
    )

    return {
        "graphs_csv": graphs_path,
        "instances_csv": instances_path,
        "calibration_curves_csv": calibration_path,
        "selected_hyperparameters_csv": selected_path,
        "benchmark_results_csv": benchmark_path,
    }


# =============================================================================
# Example use
# =============================================================================
#
# import importlib
# import real_world_benchmark as rwb
# importlib.reload(rwb)
#
# parameter_grids = rwb.build_default_parameter_grids(
#     T_values=range(1, 21),
#     layer_values=range(0, 6),
# )
#
# paths = rwb.run_real_world_benchmark(
#     load_dataset=load_dataset,
#     make_base_probabilities=make_base_probabilities,
#     calibrate_secondary_activation_grid=calibrate_secondary_activation_grid,
#     optimized_independent_cascade=optimized_independent_cascade,
#     run_method_fixed=run_method_fixed,
#     evaluate_predictions=evaluate_predictions,
#     edge_probability_summary=edge_probability_summary,
#     parameter_grids=parameter_grids,
# )
