
"""
Final synthetic benchmark for IC approximation methods.

Design
------
Core graph families:
    ErdosRenyi
    BarabasiAlbert
    WattsStrogatz
    StochasticBlockModel
    HolmeKim

n = 5000
target mean degree / IC arcs-per-node = 4, 10, 20

Edge shapes:
    constant
    weighted_cascade
    heterogeneous_dirichlet

Seed fractions:
    0.01, 0.05, 0.10, 0.20

Target secondary activation:
    g = 0.10, 0.35, 0.60

Benchmark protocol
------------------
For every overall setting

    (graph family, density, edge model, seed fraction, target g)

we generate:
    K_cal independent calibration graph/diffusion instances
    K_test independent held-out graph/diffusion instances

For each approximation method:
    1. Evaluate every candidate parameter combination on calibration instances.
    2. Select ONE parameter combination using mean calibration RMSE.
    3. Freeze it.
    4. Evaluate it on held-out instances.

The full calibration curve is saved so that parameter selection remains auditable.

Expected external callbacks
---------------------------
ICGraph:
    user's compact graph class.

make_base_probabilities(graph, edge_model=..., heterogeneous_seed=...,
                        dirichlet_concentration=...):
    from ic_calibration.py.

calibrate_secondary_activation_grid(...):
    from ic_calibration.py.

optimized_independent_cascade(graph, prior_probs, num_sim, seed=...):
    MC truth.

run_method_fixed(method, graph, prior_probs, params):
    YOUR small adapter. Must run one method at FIXED parameters.
    Accepted returns:
        predictions
    or
        (predictions, runtime_sec)

evaluate_predictions(predictions, true_probs):
    returns metric dict, preferably containing RMSE, MAE, Pearson, Spearman, AUC.

edge_probability_summary(edge_probs):
    from ic_calibration.py.

Optional:
    extra_graph_features_fn(graph, metadata)
    graph_precompute_fn(graph, metadata)
"""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx



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
# Default method grids
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


# =============================================================================
# CSV schemas
# =============================================================================

GRAPH_FIELDS = [
    "graph_id",
    "split",
    "graph_family",
    "target_mean_degree",
    "realized_mean_degree",
    "graph_rep",
    "graph_seed",
    "n",
    "m",
    "m_per_n",
    "memory_gb",
    "graph_preprocessing_time_sec",
    "generator_params_json",
    "graph_features_json",
]

INSTANCE_FIELDS = [
    "instance_id",
    "setting_id",
    "graph_id",
    "split",
    "graph_family",
    "target_mean_degree",
    "realized_mean_degree",
    "graph_rep",
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
# Graph generation
# =============================================================================

def networkx_to_icgraph(
    G,
    ICGraph,
    *,
    prob_dtype=np.float32,
):
    """
    Convert an undirected NetworkX graph to a bidirectional ICGraph.
    """
    n = int(G.number_of_nodes())

    if set(G.nodes()) != set(range(n)):
        mapping = {
            u: i
            for i, u in enumerate(G.nodes())
        }
        G = nx.relabel_nodes(
            G,
            mapping,
            copy=True,
        )

    edges = np.asarray(
        list(G.edges()),
        dtype=np.int64,
    )

    if edges.size == 0:
        src = np.empty(0, dtype=np.int32)
        dst = np.empty(0, dtype=np.int32)
    else:
        u = edges[:, 0]
        v = edges[:, 1]

        mask = u != v
        u = u[mask]
        v = v[mask]

        src = np.concatenate(
            [u, v]
        )
        dst = np.concatenate(
            [v, u]
        )

    node_dtype = (
        np.int32
        if n <= np.iinfo(np.int32).max
        else np.int64
    )

    src = src.astype(
        node_dtype,
        copy=False,
    )
    dst = dst.astype(
        node_dtype,
        copy=False,
    )

    prob = np.ones(
        len(src),
        dtype=prob_dtype,
    )

    return ICGraph.from_edges(
        n,
        src,
        dst,
        prob,
        prob_dtype=prob_dtype,
    )


def generate_benchmark_graph(
    *,
    graph_family,
    n,
    target_mean_degree,
    graph_seed,
    ICGraph,
    ws_beta=0.10,
    sbm_mu=0.30,
    sbm_num_blocks=4,
    holme_kim_triad_prob=0.50,
    prob_dtype=np.float32,
):
    """
    Generate one undirected synthetic graph and return it as bidirectional arcs.

    target_mean_degree refers to the undirected mean degree, equivalently m/n
    after expanding each undirected edge to two IC arcs.
    """
    n = int(n)
    d = float(target_mean_degree)
    seed = int(graph_seed)

    if graph_family == "ErdosRenyi":
        p = d / (n - 1)

        G = nx.fast_gnp_random_graph(
            n,
            p,
            seed=seed,
            directed=False,
        )

        generator_params = {
            "p": p,
        }

    elif graph_family == "BarabasiAlbert":
        m_attach = int(round(d / 2.0))

        if m_attach < 1:
            raise ValueError(
                "BA requires target mean degree >= 2."
            )

        G = nx.barabasi_albert_graph(
            n,
            m_attach,
            seed=seed,
        )

        generator_params = {
            "m": m_attach,
        }

    elif graph_family == "WattsStrogatz":
        k = int(round(d))

        if k % 2 != 0:
            raise ValueError(
                "Watts-Strogatz k should be even."
            )

        G = nx.watts_strogatz_graph(
            n,
            k,
            float(ws_beta),
            seed=seed,
        )

        generator_params = {
            "k": k,
            "beta": float(ws_beta),
        }

    elif graph_family == "StochasticBlockModel":
        B = int(sbm_num_blocks)

        if n % B != 0:
            raise ValueError(
                "n must be divisible by sbm_num_blocks."
            )

        block_size = n // B
        mu = float(sbm_mu)

        expected_internal_degree = (
            1.0 - mu
        ) * d
        expected_external_degree = (
            mu * d
        )

        p_in = (
            expected_internal_degree
            / (block_size - 1)
        )

        p_out = (
            expected_external_degree
            / (n - block_size)
        )

        P = np.full(
            (B, B),
            p_out,
            dtype=float,
        )
        np.fill_diagonal(
            P,
            p_in,
        )

        G = nx.stochastic_block_model(
            [block_size] * B,
            P.tolist(),
            seed=seed,
            directed=False,
            selfloops=False,
        )

        generator_params = {
            "num_blocks": B,
            "block_size": block_size,
            "mu": mu,
            "p_in": p_in,
            "p_out": p_out,
        }

    elif graph_family == "HolmeKim":
        m_attach = int(round(d / 2.0))

        if m_attach < 1:
            raise ValueError(
                "Holme-Kim requires target mean degree >= 2."
            )

        G = nx.powerlaw_cluster_graph(
            n,
            m_attach,
            float(holme_kim_triad_prob),
            seed=seed,
        )

        generator_params = {
            "m": m_attach,
            "triad_probability": float(
                holme_kim_triad_prob
            ),
        }

    else:
        raise ValueError(
            f"Unknown graph family {graph_family!r}."
        )

    graph = networkx_to_icgraph(
        G,
        ICGraph,
        prob_dtype=prob_dtype,
    )

    metadata = {
        "graph_family": graph_family,
        "target_mean_degree": d,
        "realized_mean_degree": (
            float(graph.m) / float(graph.n)
        ),
        "generator_params": generator_params,
    }

    return graph, metadata


# =============================================================================
# Helpers
# =============================================================================

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


def _memory_gb(graph):
    if hasattr(
        graph,
        "memory_gb",
    ):
        try:
            return float(
                graph.memory_gb()
            )
        except Exception:
            pass

    total = 0

    for name in (
        "src",
        "dst",
        "prob",
    ):
        arr = getattr(
            graph,
            name,
            None,
        )

        if isinstance(
            arr,
            np.ndarray,
        ):
            total += arr.nbytes

    return total / (1024 ** 3)


def _select_params_for_setting(
    records,
    *,
    parameter_grid,
    selection_tolerance=0.0,
):
    """
    Select parameters by mean calibration RMSE.

    selection_tolerance is ADDITIVE:
        eligible if mean_RMSE <= best_mean_RMSE + tolerance.

    Within the eligible set, choose the cheaper/smaller parameter combination:
        smaller T first,
        then smaller layers.

    With tolerance=0 this is ordinary calibration argmin with deterministic
    tie-breaking.
    """
    df = pd.DataFrame(
        records
    )

    df = df[
        (df["status"] == "ok")
        & pd.to_numeric(
            df["RMSE"],
            errors="coerce",
        ).notna()
    ].copy()

    if df.empty:
        return None, None

    grouped = (
        df.groupby(
            "params_json",
            as_index=False,
        )
        .agg(
            mean_RMSE=(
                "RMSE",
                "mean",
            ),
            median_RMSE=(
                "RMSE",
                "median",
            ),
            std_RMSE=(
                "RMSE",
                "std",
            ),
            mean_runtime_sec=(
                "runtime_sec",
                "mean",
            ),
            n_calibration_instances=(
                "instance_id",
                "nunique",
            ),
        )
    )

    best = float(
        grouped["mean_RMSE"].min()
    )

    eligible = grouped[
        grouped["mean_RMSE"]
        <= best
        + float(selection_tolerance)
        + 1e-15
    ].copy()

    params_lookup = {
        _params_json(p): dict(p)
        for p in parameter_grid
    }

    def tie_key(row):
        params = params_lookup[
            row["params_json"]
        ]

        T = params.get(
            "T",
            -1,
        )
        L = params.get(
            "layers",
            -1,
        )

        # If a method has no T, it naturally ties at -1.
        return (
            float(T)
            if T != ""
            else -1,
            float(L)
            if L != ""
            else -1,
            float(
                row["mean_RMSE"]
            ),
        )

    chosen_index = min(
        eligible.index,
        key=lambda idx: tie_key(
            eligible.loc[idx]
        ),
    )

    chosen = eligible.loc[
        chosen_index
    ]

    params = params_lookup[
        chosen["params_json"]
    ]

    selection_row = {
        **_param_fields(
            params
        ),
        "mean_RMSE": float(
            chosen["mean_RMSE"]
        ),
        "median_RMSE": float(
            chosen["median_RMSE"]
        ),
        "std_RMSE": float(
            chosen["std_RMSE"]
        )
        if pd.notna(
            chosen["std_RMSE"]
        )
        else np.nan,
        "mean_runtime_sec": float(
            chosen["mean_runtime_sec"]
        ),
        "n_calibration_instances": int(
            chosen[
                "n_calibration_instances"
            ]
        ),
        "selection_tolerance": float(
            selection_tolerance
        ),
        "best_mean_RMSE": best,
    }

    return params, selection_row


# =============================================================================
# Instance creation + MC
# =============================================================================

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
# Benchmark runner
# =============================================================================

def run_synthetic_benchmark(
    *,
    ICGraph,
    make_base_probabilities,
    calibrate_secondary_activation_grid,
    optimized_independent_cascade,
    run_method_fixed,
    evaluate_predictions,
    edge_probability_summary,
    output_dir="results/synthetic_benchmark",
    methods=DEFAULT_METHODS,
    parameter_grids=None,
    base_seed=42,
    n=5000,
    graph_families=(
        "ErdosRenyi",
        "BarabasiAlbert",
        "WattsStrogatz",
        "StochasticBlockModel",
        "HolmeKim",
    ),
    mean_degrees=(
        4,
        10,
        20,
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
    calibration_num_sim=10_000,
    test_num_sim=50_000,
    pilot_num_sim=250,
    pilot_repeats=1,
    binary_search_steps=10,
    target_tol=0.02,
    selection_tolerance=0.0,
    ws_beta=0.10,
    sbm_mu=0.30,
    sbm_num_blocks=4,
    holme_kim_triad_prob=0.50,
    prob_dtype=np.float32,
    extra_graph_features_fn=None,
    graph_precompute_fn=None,
    resume=True,
):
    """
    Run final synthetic benchmark with calibration/test separation.

    IMPORTANT
    ---------
    Calibration and held-out instances use DIFFERENT graph realizations.
    Therefore selected method hyperparameters must generalize to a fresh graph
    drawn from the same graph-family / density / diffusion regime.

    If you instead want to hold graph topology fixed and vary only edge/seed
    realizations, this can be changed easily, but the default here is the
    stronger synthetic generalization test.
    """
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if parameter_grids is None:
        parameter_grids = (
            build_default_parameter_grids(
                methods=methods
            )
        )

    graphs_path = (
        output_dir
        / "graphs.csv"
    )
    instances_path = (
        output_dir
        / "instances.csv"
    )
    calibration_path = (
        output_dir
        / "calibration_curves.csv"
    )
    selected_path = (
        output_dir
        / "selected_hyperparameters.csv"
    )
    benchmark_path = (
        output_dir
        / "benchmark_results.csv"
    )

    graph_writer = CsvAppender(
        graphs_path,
        GRAPH_FIELDS,
    )
    instance_writer = CsvAppender(
        instances_path,
        INSTANCE_FIELDS,
    )
    calibration_writer = (
        CsvAppender(
            calibration_path,
            CALIBRATION_FIELDS,
        )
    )
    selected_writer = (
        CsvAppender(
            selected_path,
            SELECTED_FIELDS,
        )
    )
    benchmark_writer = (
        CsvAppender(
            benchmark_path,
            BENCHMARK_FIELDS,
        )
    )

    calibration_done = (
        _existing_key_set(
            calibration_path,
            [
                "instance_id",
                "method",
                "params_json",
            ],
        )
        if resume
        else set()
    )

    benchmark_done = (
        _existing_key_set(
            benchmark_path,
            [
                "instance_id",
                "method",
            ],
        )
        if resume
        else set()
    )

    graph_rows_done = (
        _existing_key_set(
            graphs_path,
            ["graph_id"],
        )
        if resume
        else set()
    )

    instance_rows_done = (
        _existing_key_set(
            instances_path,
            ["instance_id"],
        )
        if resume
        else set()
    )

    selected_done = (
        _existing_key_set(
            selected_path,
            [
                "setting_id",
                "method",
            ],
        )
        if resume
        else set()
    )

    family_to_idx = {
        family: i
        for i, family
        in enumerate(
            graph_families
        )
    }

    degree_to_idx = {
        float(d): i
        for i, d
        in enumerate(
            mean_degrees
        )
    }

    # -------------------------------------------------------------------------
    # Main structural loop.
    # -------------------------------------------------------------------------
    for family in graph_families:
        family_idx = (
            family_to_idx[
                family
            ]
        )

        for target_d in mean_degrees:
            target_d = float(
                target_d
            )

            degree_idx = (
                degree_to_idx[
                    target_d
                ]
            )

            print(
                f"\n{'=' * 82}\n"
                f"{family}, target mean degree={target_d:g}\n"
                f"{'=' * 82}"
            )

            # =================================================================
            # CALIBRATION INSTANCES
            # =================================================================
            calibration_records_by_setting_method = {}

            for rep in range(
                int(
                    num_calibration_instances
                )
            ):
                split = "calibration"

                graph_seed = derive_seed(
                    base_seed,
                    100,
                    family_idx,
                    degree_idx,
                    0,
                    rep,
                )

                graph_id = (
                    f"{family}"
                    f"__d{target_d:g}"
                    f"__cal"
                    f"__r{rep:02d}"
                )

                graph, graph_meta = (
                    generate_benchmark_graph(
                        graph_family=family,
                        n=n,
                        target_mean_degree=(
                            target_d
                        ),
                        graph_seed=(
                            graph_seed
                        ),
                        ICGraph=ICGraph,
                        ws_beta=ws_beta,
                        sbm_mu=sbm_mu,
                        sbm_num_blocks=(
                            sbm_num_blocks
                        ),
                        holme_kim_triad_prob=(
                            holme_kim_triad_prob
                        ),
                        prob_dtype=(
                            prob_dtype
                        ),
                    )
                )

                precompute_time = 0.0

                if (
                    graph_precompute_fn
                    is not None
                ):
                    start = (
                        time.perf_counter()
                    )

                    graph_precompute_fn(
                        graph,
                        graph_meta,
                    )

                    precompute_time = (
                        time.perf_counter()
                        - start
                    )

                graph_features = (
                    basic_graph_characteristics(
                        graph
                    )
                )

                extra_features = {}

                if (
                    extra_graph_features_fn
                    is not None
                ):
                    extra_features = dict(
                        extra_graph_features_fn(
                            graph,
                            graph_meta,
                        )
                    )

                if (
                    (graph_id,)
                    not in graph_rows_done
                ):
                    graph_writer.append(
                        {
                            "graph_id": graph_id,
                            "split": split,
                            "graph_family": family,
                            "target_mean_degree": target_d,
                            "realized_mean_degree": (
                                graph_meta[
                                    "realized_mean_degree"
                                ]
                            ),
                            "graph_rep": rep,
                            "graph_seed": graph_seed,
                            "n": graph.n,
                            "m": graph.m,
                            "m_per_n": (
                                float(graph.m)
                                / float(graph.n)
                            ),
                            "memory_gb": (
                                _memory_gb(
                                    graph
                                )
                            ),
                            "graph_preprocessing_time_sec": (
                                precompute_time
                            ),
                            "generator_params_json": (
                                json_dumps(
                                    graph_meta[
                                        "generator_params"
                                    ]
                                )
                            ),
                            "graph_features_json": (
                                json_dumps(
                                    {
                                        **graph_features,
                                        **extra_features,
                                    }
                                )
                            ),
                        }
                    )

                    graph_rows_done.add(
                        (graph_id,)
                    )

                # Same seed set reused across edge models within graph + sf.
                priors = {}

                for sf_idx, sf in enumerate(
                    seed_fractions
                ):
                    seedset_seed = derive_seed(
                        base_seed,
                        200,
                        family_idx,
                        degree_idx,
                        0,
                        rep,
                        sf_idx,
                    )

                    priors[
                        float(sf)
                    ] = (
                        seedset_seed
                    )

                for edge_idx, edge_model in enumerate(
                    edge_models
                ):
                    edge_seed = derive_seed(
                        base_seed,
                        300,
                        family_idx,
                        degree_idx,
                        0,
                        rep,
                        edge_idx,
                    )

                    for sf_idx, sf in enumerate(
                        seed_fractions
                    ):
                        seedset_seed = (
                            priors[
                                float(sf)
                            ]
                        )

                        for target_idx, target_g in enumerate(
                            target_secondary_activations
                        ):
                            target_g = float(
                                target_g
                            )

                            setting_id = (
                                f"{family}"
                                f"__d{target_d:g}"
                                f"__{edge_model}"
                                f"__s{float(sf):.3f}"
                                f"__g{target_g:.2f}"
                            )

                            instance_id = (
                                f"{setting_id}"
                                f"__cal"
                                f"__r{rep:02d}"
                            )

                            pilot_seed = (
                                derive_seed(
                                    base_seed,
                                    400,
                                    family_idx,
                                    degree_idx,
                                    0,
                                    rep,
                                    edge_idx,
                                    sf_idx,
                                    target_idx,
                                )
                            )

                            mc_seed = (
                                derive_seed(
                                    base_seed,
                                    500,
                                    family_idx,
                                    degree_idx,
                                    0,
                                    rep,
                                    edge_idx,
                                    sf_idx,
                                    target_idx,
                                )
                            )

                            print(
                                f"\n[CAL] {instance_id}"
                            )

                            try:
                                inst = (
                                    prepare_diffusion_instance(
                                        graph=graph,
                                        graph_meta=graph_meta,
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
                                        num_sim=(
                                            calibration_num_sim
                                        ),
                                        mc_seed=mc_seed,
                                        dirichlet_concentration=(
                                            dirichlet_concentration
                                        ),
                                        pilot_num_sim=(
                                            pilot_num_sim
                                        ),
                                        pilot_repeats=(
                                            pilot_repeats
                                        ),
                                        binary_search_steps=(
                                            binary_search_steps
                                        ),
                                        target_tol=(
                                            target_tol
                                        ),
                                    )
                                )

                            except Exception as exc:
                                print(
                                    f"  [INSTANCE ERROR] "
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                )

                                if (
                                    (instance_id,)
                                    not in instance_rows_done
                                ):
                                    instance_writer.append(
                                        {
                                            "instance_id": instance_id,
                                            "setting_id": setting_id,
                                            "graph_id": graph_id,
                                            "split": split,
                                            "graph_family": family,
                                            "target_mean_degree": target_d,
                                            "realized_mean_degree": (
                                                graph_meta[
                                                    "realized_mean_degree"
                                                ]
                                            ),
                                            "graph_rep": rep,
                                            "edge_model": edge_model,
                                            "edge_seed": edge_seed,
                                            "dirichlet_concentration": (
                                                dirichlet_concentration
                                            ),
                                            "requested_seed_fraction": (
                                                sf
                                            ),
                                            "seedset_seed": seedset_seed,
                                            "target_secondary_activation": (
                                                target_g
                                            ),
                                            "pilot_seed": pilot_seed,
                                            "mc_num_sim": (
                                                calibration_num_sim
                                            ),
                                            "mc_seed": mc_seed,
                                            "status": "error",
                                            "error_type": (
                                                type(exc).__name__
                                            ),
                                            "error_message": (
                                                str(exc)[:1000]
                                            ),
                                        }
                                    )

                                    instance_rows_done.add(
                                        (instance_id,)
                                    )

                                continue

                            info = (
                                inst[
                                    "calibration_info"
                                ]
                            )
                            es = (
                                inst[
                                    "edge_summary"
                                ]
                            )
                            ts = (
                                inst[
                                    "true_summary"
                                ]
                            )
                            seed_features = (
                                inst[
                                    "seed_features"
                                ]
                            )

                            if (
                                (instance_id,)
                                not in instance_rows_done
                            ):
                                instance_writer.append(
                                    {
                                        "instance_id": instance_id,
                                        "setting_id": setting_id,
                                        "graph_id": graph_id,
                                        "split": split,
                                        "graph_family": family,
                                        "target_mean_degree": target_d,
                                        "realized_mean_degree": (
                                            graph_meta[
                                                "realized_mean_degree"
                                            ]
                                        ),
                                        "graph_rep": rep,
                                        "edge_model": edge_model,
                                        "edge_seed": edge_seed,
                                        "dirichlet_concentration": (
                                            dirichlet_concentration
                                        ),
                                        "requested_seed_fraction": sf,
                                        "realized_seed_fraction": (
                                            len(
                                                inst[
                                                    "selected"
                                                ]
                                            )
                                            / graph.n
                                        ),
                                        "seed_count": (
                                            len(
                                                inst[
                                                    "selected"
                                                ]
                                            )
                                        ),
                                        "seedset_seed": seedset_seed,
                                        "target_secondary_activation": target_g,
                                        "pilot_seed": pilot_seed,
                                        "alpha": info.get(
                                            "alpha",
                                            "",
                                        ),
                                        "pilot_achieved_secondary_activation": (
                                            info.get(
                                                "achieved_secondary_activation",
                                                "",
                                            )
                                        ),
                                        "pilot_secondary_activation_abs_error": (
                                            info.get(
                                                "secondary_activation_abs_error",
                                                "",
                                            )
                                        ),
                                        "calibration_clipped_fraction": (
                                            info.get(
                                                "clipped_fraction",
                                                "",
                                            )
                                        ),
                                        "edge_prob_mean": es.get(
                                            "edge_prob_mean",
                                            es.get(
                                                "mean",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_std": es.get(
                                            "edge_prob_std",
                                            es.get(
                                                "std",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q50": es.get(
                                            "edge_prob_q50",
                                            es.get(
                                                "q50",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q90": es.get(
                                            "edge_prob_q90",
                                            es.get(
                                                "q90",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q99": es.get(
                                            "edge_prob_q99",
                                            es.get(
                                                "q99",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_max": es.get(
                                            "edge_prob_max",
                                            es.get(
                                                "max",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_frac_ge_0_1": es.get(
                                            "edge_prob_frac_ge_0_1",
                                            "",
                                        ),
                                        "edge_prob_frac_ge_0_25": es.get(
                                            "edge_prob_frac_ge_0_25",
                                            "",
                                        ),
                                        "edge_prob_frac_ge_0_5": es.get(
                                            "edge_prob_frac_ge_0_5",
                                            "",
                                        ),
                                        "true_mean": ts.get(
                                            "true_mean",
                                            "",
                                        ),
                                        "true_std": ts.get(
                                            "true_std",
                                            "",
                                        ),
                                        "true_q10": ts.get(
                                            "true_q10",
                                            "",
                                        ),
                                        "true_q50": ts.get(
                                            "true_q50",
                                            "",
                                        ),
                                        "true_q90": ts.get(
                                            "true_q90",
                                            "",
                                        ),
                                        "true_secondary_activation": (
                                            inst[
                                                "true_secondary_activation"
                                            ]
                                        ),
                                        "mc_num_sim": (
                                            calibration_num_sim
                                        ),
                                        "mc_seed": mc_seed,
                                        "mc_runtime_sec": (
                                            inst[
                                                "mc_runtime_sec"
                                            ]
                                        ),
                                        "seed_features_json": (
                                            json_dumps(
                                                seed_features
                                            )
                                        ),
                                        "calibration_info_json": (
                                            json_dumps(
                                                info
                                            )
                                        ),
                                        "status": "ok",
                                        "error_type": "",
                                        "error_message": "",
                                    }
                                )

                                instance_rows_done.add(
                                    (instance_id,)
                                )

                            # -------------------------------------------------
                            # Full parameter curves on calibration data.
                            # -------------------------------------------------
                            for method in methods:
                                parameter_grid = (
                                    parameter_grids[
                                        method
                                    ]
                                )

                                key = (
                                    setting_id,
                                    method,
                                )

                                calibration_records_by_setting_method.setdefault(
                                    key,
                                    [],
                                )

                                for params in parameter_grid:
                                    params_json = (
                                        _params_json(
                                            params
                                        )
                                    )

                                    done_key = (
                                        instance_id,
                                        method,
                                        params_json,
                                    )

                                    if (
                                        done_key
                                        in calibration_done
                                    ):
                                        # Existing rows will be loaded below
                                        # before selection.
                                        continue

                                    try:
                                        preds, runtime = (
                                            _run_fixed(
                                                run_method_fixed,
                                                method,
                                                inst["graph"],
                                                inst["prior_probs"],
                                                params,
                                            )
                                        )

                                        preds = probability_array(
                                            preds,
                                            graph.n,
                                        )

                                        metrics = _safe_metric_dict(
                                            evaluate_predictions,
                                            preds,
                                            inst["true_probs"],
                                        )

                                        row = {
                                            "setting_id": setting_id,
                                            "instance_id": instance_id,
                                            "method": method,
                                            **_param_fields(
                                                params
                                            ),
                                            **metrics,
                                            "runtime_sec": runtime,
                                            "status": "ok",
                                            "error_type": "",
                                            "error_message": "",
                                        }

                                    except Exception as exc:
                                        row = {
                                            "setting_id": setting_id,
                                            "instance_id": instance_id,
                                            "method": method,
                                            **_param_fields(
                                                params
                                            ),
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

                                        print(
                                            f"  [METHOD ERROR] "
                                            f"{method} {params}: "
                                            f"{type(exc).__name__}: "
                                            f"{exc}"
                                        )

                                    calibration_writer.append(
                                        row
                                    )

                                    calibration_done.add(
                                        done_key
                                    )

                                    calibration_records_by_setting_method[
                                        key
                                    ].append(
                                        row
                                    )

            # -----------------------------------------------------------------
            # Reload ALL calibration records for this family/density, including
            # rows from previous/resumed execution.
            # -----------------------------------------------------------------
            if calibration_path.exists():
                cal_df = pd.read_csv(
                    calibration_path
                )

                prefix = (
                    f"{family}"
                    f"__d{target_d:g}"
                    f"__"
                )

                cal_df = cal_df[
                    cal_df[
                        "setting_id"
                    ].astype(str).str.startswith(
                        prefix
                    )
                ].copy()
            else:
                cal_df = pd.DataFrame()

            # =================================================================
            # SELECT FROZEN PARAMETERS
            # =================================================================
            selected_params = {}

            for edge_model in edge_models:
                for sf in seed_fractions:
                    for target_g in target_secondary_activations:
                        setting_id = (
                            f"{family}"
                            f"__d{target_d:g}"
                            f"__{edge_model}"
                            f"__s{float(sf):.3f}"
                            f"__g{float(target_g):.2f}"
                        )

                        for method in methods:
                            key = (
                                setting_id,
                                method,
                            )

                            rows = (
                                cal_df[
                                    (
                                        cal_df[
                                            "setting_id"
                                        ]
                                        == setting_id
                                    )
                                    & (
                                        cal_df[
                                            "method"
                                        ]
                                        == method
                                    )
                                ]
                                .to_dict(
                                    "records"
                                )
                                if not cal_df.empty
                                else []
                            )

                            params, selection = (
                                _select_params_for_setting(
                                    rows,
                                    parameter_grid=(
                                        parameter_grids[
                                            method
                                        ]
                                    ),
                                    selection_tolerance=(
                                        selection_tolerance
                                    ),
                                )
                            )

                            if params is None:
                                print(
                                    f"[NO PARAM SELECTION] "
                                    f"{setting_id}, {method}"
                                )
                                continue

                            selected_params[
                                key
                            ] = params

                            if (
                                key
                                not in selected_done
                            ):
                                selected_writer.append(
                                    {
                                        "setting_id": setting_id,
                                        "method": method,
                                        **selection,
                                    }
                                )

                                selected_done.add(
                                    key
                                )

            # =================================================================
            # HELD-OUT TEST INSTANCES
            # =================================================================
            for rep in range(
                int(
                    num_test_instances
                )
            ):
                split = "test"

                graph_seed = derive_seed(
                    base_seed,
                    100,
                    family_idx,
                    degree_idx,
                    1,
                    rep,
                )

                graph_id = (
                    f"{family}"
                    f"__d{target_d:g}"
                    f"__test"
                    f"__r{rep:02d}"
                )

                graph, graph_meta = (
                    generate_benchmark_graph(
                        graph_family=family,
                        n=n,
                        target_mean_degree=target_d,
                        graph_seed=graph_seed,
                        ICGraph=ICGraph,
                        ws_beta=ws_beta,
                        sbm_mu=sbm_mu,
                        sbm_num_blocks=(
                            sbm_num_blocks
                        ),
                        holme_kim_triad_prob=(
                            holme_kim_triad_prob
                        ),
                        prob_dtype=(
                            prob_dtype
                        ),
                    )
                )

                precompute_time = 0.0

                if graph_precompute_fn is not None:
                    start = time.perf_counter()

                    graph_precompute_fn(
                        graph,
                        graph_meta,
                    )

                    precompute_time = (
                        time.perf_counter()
                        - start
                    )

                graph_features = (
                    basic_graph_characteristics(
                        graph
                    )
                )

                extra_features = {}

                if extra_graph_features_fn is not None:
                    extra_features = dict(
                        extra_graph_features_fn(
                            graph,
                            graph_meta,
                        )
                    )

                if (
                    (graph_id,)
                    not in graph_rows_done
                ):
                    graph_writer.append(
                        {
                            "graph_id": graph_id,
                            "split": split,
                            "graph_family": family,
                            "target_mean_degree": target_d,
                            "realized_mean_degree": (
                                graph_meta[
                                    "realized_mean_degree"
                                ]
                            ),
                            "graph_rep": rep,
                            "graph_seed": graph_seed,
                            "n": graph.n,
                            "m": graph.m,
                            "m_per_n": (
                                float(graph.m)
                                / float(graph.n)
                            ),
                            "memory_gb": (
                                _memory_gb(
                                    graph
                                )
                            ),
                            "graph_preprocessing_time_sec": (
                                precompute_time
                            ),
                            "generator_params_json": (
                                json_dumps(
                                    graph_meta[
                                        "generator_params"
                                    ]
                                )
                            ),
                            "graph_features_json": (
                                json_dumps(
                                    {
                                        **graph_features,
                                        **extra_features,
                                    }
                                )
                            ),
                        }
                    )

                    graph_rows_done.add(
                        (graph_id,)
                    )

                priors = {}

                for sf_idx, sf in enumerate(
                    seed_fractions
                ):
                    seedset_seed = derive_seed(
                        base_seed,
                        200,
                        family_idx,
                        degree_idx,
                        1,
                        rep,
                        sf_idx,
                    )

                    priors[
                        float(sf)
                    ] = seedset_seed

                for edge_idx, edge_model in enumerate(
                    edge_models
                ):
                    edge_seed = derive_seed(
                        base_seed,
                        300,
                        family_idx,
                        degree_idx,
                        1,
                        rep,
                        edge_idx,
                    )

                    for sf_idx, sf in enumerate(
                        seed_fractions
                    ):
                        seedset_seed = (
                            priors[
                                float(sf)
                            ]
                        )

                        for target_idx, target_g in enumerate(
                            target_secondary_activations
                        ):
                            target_g = float(
                                target_g
                            )

                            setting_id = (
                                f"{family}"
                                f"__d{target_d:g}"
                                f"__{edge_model}"
                                f"__s{float(sf):.3f}"
                                f"__g{target_g:.2f}"
                            )

                            instance_id = (
                                f"{setting_id}"
                                f"__test"
                                f"__r{rep:02d}"
                            )

                            # If every method already has this test row, skip
                            # rebuilding MC truth entirely.
                            expected_test_keys = [
                                (
                                    instance_id,
                                    method,
                                )
                                for method in methods
                                if (
                                    setting_id,
                                    method,
                                )
                                in selected_params
                            ]

                            if (
                                expected_test_keys
                                and all(
                                    key
                                    in benchmark_done
                                    for key
                                    in expected_test_keys
                                )
                            ):
                                continue

                            pilot_seed = derive_seed(
                                base_seed,
                                400,
                                family_idx,
                                degree_idx,
                                1,
                                rep,
                                edge_idx,
                                sf_idx,
                                target_idx,
                            )

                            mc_seed = derive_seed(
                                base_seed,
                                500,
                                family_idx,
                                degree_idx,
                                1,
                                rep,
                                edge_idx,
                                sf_idx,
                                target_idx,
                            )

                            print(
                                f"\n[TEST] {instance_id}"
                            )

                            try:
                                inst = (
                                    prepare_diffusion_instance(
                                        graph=graph,
                                        graph_meta=graph_meta,
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
                                        num_sim=(
                                            test_num_sim
                                        ),
                                        mc_seed=mc_seed,
                                        dirichlet_concentration=(
                                            dirichlet_concentration
                                        ),
                                        pilot_num_sim=(
                                            pilot_num_sim
                                        ),
                                        pilot_repeats=(
                                            pilot_repeats
                                        ),
                                        binary_search_steps=(
                                            binary_search_steps
                                        ),
                                        target_tol=(
                                            target_tol
                                        ),
                                    )
                                )

                            except Exception as exc:
                                print(
                                    f"  [INSTANCE ERROR] "
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                )

                                if (
                                    (instance_id,)
                                    not in instance_rows_done
                                ):
                                    instance_writer.append(
                                        {
                                            "instance_id": instance_id,
                                            "setting_id": setting_id,
                                            "graph_id": graph_id,
                                            "split": split,
                                            "graph_family": family,
                                            "target_mean_degree": target_d,
                                            "realized_mean_degree": (
                                                graph_meta[
                                                    "realized_mean_degree"
                                                ]
                                            ),
                                            "graph_rep": rep,
                                            "edge_model": edge_model,
                                            "edge_seed": edge_seed,
                                            "dirichlet_concentration": (
                                                dirichlet_concentration
                                            ),
                                            "requested_seed_fraction": sf,
                                            "seedset_seed": seedset_seed,
                                            "target_secondary_activation": target_g,
                                            "pilot_seed": pilot_seed,
                                            "mc_num_sim": (
                                                test_num_sim
                                            ),
                                            "mc_seed": mc_seed,
                                            "status": "error",
                                            "error_type": (
                                                type(exc).__name__
                                            ),
                                            "error_message": (
                                                str(exc)[:1000]
                                            ),
                                        }
                                    )

                                    instance_rows_done.add(
                                        (instance_id,)
                                    )

                                continue

                            info = inst[
                                "calibration_info"
                            ]
                            es = inst[
                                "edge_summary"
                            ]
                            ts = inst[
                                "true_summary"
                            ]

                            if (
                                (instance_id,)
                                not in instance_rows_done
                            ):
                                instance_writer.append(
                                    {
                                        "instance_id": instance_id,
                                        "setting_id": setting_id,
                                        "graph_id": graph_id,
                                        "split": split,
                                        "graph_family": family,
                                        "target_mean_degree": target_d,
                                        "realized_mean_degree": (
                                            graph_meta[
                                                "realized_mean_degree"
                                            ]
                                        ),
                                        "graph_rep": rep,
                                        "edge_model": edge_model,
                                        "edge_seed": edge_seed,
                                        "dirichlet_concentration": (
                                            dirichlet_concentration
                                        ),
                                        "requested_seed_fraction": sf,
                                        "realized_seed_fraction": (
                                            len(
                                                inst[
                                                    "selected"
                                                ]
                                            )
                                            / graph.n
                                        ),
                                        "seed_count": len(
                                            inst[
                                                "selected"
                                            ]
                                        ),
                                        "seedset_seed": seedset_seed,
                                        "target_secondary_activation": target_g,
                                        "pilot_seed": pilot_seed,
                                        "alpha": info.get(
                                            "alpha",
                                            "",
                                        ),
                                        "pilot_achieved_secondary_activation": (
                                            info.get(
                                                "achieved_secondary_activation",
                                                "",
                                            )
                                        ),
                                        "pilot_secondary_activation_abs_error": (
                                            info.get(
                                                "secondary_activation_abs_error",
                                                "",
                                            )
                                        ),
                                        "calibration_clipped_fraction": (
                                            info.get(
                                                "clipped_fraction",
                                                "",
                                            )
                                        ),
                                        "edge_prob_mean": es.get(
                                            "edge_prob_mean",
                                            es.get(
                                                "mean",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_std": es.get(
                                            "edge_prob_std",
                                            es.get(
                                                "std",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q50": es.get(
                                            "edge_prob_q50",
                                            es.get(
                                                "q50",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q90": es.get(
                                            "edge_prob_q90",
                                            es.get(
                                                "q90",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_q99": es.get(
                                            "edge_prob_q99",
                                            es.get(
                                                "q99",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_max": es.get(
                                            "edge_prob_max",
                                            es.get(
                                                "max",
                                                "",
                                            ),
                                        ),
                                        "edge_prob_frac_ge_0_1": es.get(
                                            "edge_prob_frac_ge_0_1",
                                            "",
                                        ),
                                        "edge_prob_frac_ge_0_25": es.get(
                                            "edge_prob_frac_ge_0_25",
                                            "",
                                        ),
                                        "edge_prob_frac_ge_0_5": es.get(
                                            "edge_prob_frac_ge_0_5",
                                            "",
                                        ),
                                        "true_mean": ts.get(
                                            "true_mean",
                                            "",
                                        ),
                                        "true_std": ts.get(
                                            "true_std",
                                            "",
                                        ),
                                        "true_q10": ts.get(
                                            "true_q10",
                                            "",
                                        ),
                                        "true_q50": ts.get(
                                            "true_q50",
                                            "",
                                        ),
                                        "true_q90": ts.get(
                                            "true_q90",
                                            "",
                                        ),
                                        "true_secondary_activation": (
                                            inst[
                                                "true_secondary_activation"
                                            ]
                                        ),
                                        "mc_num_sim": (
                                            test_num_sim
                                        ),
                                        "mc_seed": mc_seed,
                                        "mc_runtime_sec": (
                                            inst[
                                                "mc_runtime_sec"
                                            ]
                                        ),
                                        "seed_features_json": (
                                            json_dumps(
                                                inst[
                                                    "seed_features"
                                                ]
                                            )
                                        ),
                                        "calibration_info_json": (
                                            json_dumps(
                                                info
                                            )
                                        ),
                                        "status": "ok",
                                        "error_type": "",
                                        "error_message": "",
                                    }
                                )

                                instance_rows_done.add(
                                    (instance_id,)
                                )

                            # -------------------------------------------------
                            # Frozen benchmark evaluation.
                            # -------------------------------------------------
                            for method in methods:
                                selected_key = (
                                    setting_id,
                                    method,
                                )

                                if (
                                    selected_key
                                    not in selected_params
                                ):
                                    continue

                                done_key = (
                                    instance_id,
                                    method,
                                )

                                if (
                                    done_key
                                    in benchmark_done
                                ):
                                    continue

                                params = (
                                    selected_params[
                                        selected_key
                                    ]
                                )

                                try:
                                    preds, runtime = (
                                        _run_fixed(
                                            run_method_fixed,
                                            method,
                                            inst["graph"],
                                            inst["prior_probs"],
                                            params,
                                        )
                                    )

                                    preds = (
                                        probability_array(
                                            preds,
                                            graph.n,
                                        )
                                    )

                                    metrics = (
                                        _safe_metric_dict(
                                            evaluate_predictions,
                                            preds,
                                            inst["true_probs"],
                                        )
                                    )

                                    row = {
                                        "setting_id": setting_id,
                                        "instance_id": instance_id,
                                        "method": method,
                                        **_param_fields(
                                            params
                                        ),
                                        **metrics,
                                        "runtime_sec": runtime,
                                        "status": "ok",
                                        "error_type": "",
                                        "error_message": "",
                                    }

                                except Exception as exc:
                                    row = {
                                        "setting_id": setting_id,
                                        "instance_id": instance_id,
                                        "method": method,
                                        **_param_fields(
                                            params
                                        ),
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

                                    print(
                                        f"  [METHOD ERROR] "
                                        f"{method} {params}: "
                                        f"{type(exc).__name__}: "
                                        f"{exc}"
                                    )

                                benchmark_writer.append(
                                    row
                                )

                                benchmark_done.add(
                                    done_key
                                )

    print(
        "\nSynthetic benchmark finished.\n"
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
# Example adapter skeleton
# =============================================================================
#
# The benchmark deliberately does not guess the exact names/signatures of your
# implementation functions. Define this in your notebook/project:
#
# def run_method_fixed(method, graph, prior_probs, params):
#     T = params.get("T")
#     layers = params.get("layers")
#
#     if method == "dmp_est":
#         return dmp_est(graph, prior_probs, T=T)
#
#     if method == "dmp_est_r2":
#         return dmp_est_r2(graph, prior_probs, T=T)
#
#     if method == "swe":
#         return swe(graph, prior_probs, T=T)
#
#     ...
#
#     if method == "SPM":
#         return SPM(graph, prior_probs)
#
#     raise ValueError(method)
#
#
# Then:
#
# parameter_grids = build_default_parameter_grids(
#     T_values=range(1, 21),
#     layer_values=range(0, 6),
# )
#
# paths = run_synthetic_benchmark(
#     ICGraph=ICGraph,
#     make_base_probabilities=make_base_probabilities,
#     calibrate_secondary_activation_grid=calibrate_secondary_activation_grid,
#     optimized_independent_cascade=optimized_independent_cascade,
#     run_method_fixed=run_method_fixed,
#     evaluate_predictions=evaluate_predictions,
#     edge_probability_summary=edge_probability_summary,
#     parameter_grids=parameter_grids,
# )
