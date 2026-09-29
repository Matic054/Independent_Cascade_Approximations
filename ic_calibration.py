"""
IC regime calibration for exploratory Independent Cascade experiments.

Recommended experimental pipeline
---------------------------------

    graph topology
        -> edge-probability SHAPE w_e
        -> seed prior p0
        -> scalar edge-strength calibration alpha
        -> p_e(alpha) = min(max_prob, alpha * w_e)

For a fixed graph, fixed edge-probability shape, and fixed seed prior, alpha is
chosen so that a cheap pilot Monte Carlo estimate reaches a target SECONDARY
activation fraction

    g = (mean(final_activation) - mean(p0)) / (1 - mean(p0)).

Thus g measures the fraction of initially inactive probability mass activated
by diffusion. This is preferable to calibrating total final activation because
it does not count the initial seed mass itself as diffusion.

Pilot evaluator interface
-------------------------
The calibrator is deliberately independent of a particular Monte Carlo
implementation. Supply a callable

    pilot_evaluator(edge_prob, num_sim, seed) -> result

where result may be either:
    - a scalar mean final activation,
    - a length-n array of final node probabilities,
    - a dict whose values are final node probabilities.

A typical wrapper around an existing IC Monte Carlo function is:

    def pilot_eval(edge_prob, num_sim, seed):
        out = optimized_independent_cascade(
            graph,
            prior_probs,
            edge_prob,
            num_sim,
            seed=seed,
        )
        return out

The same pilot seeds are reused at every candidate alpha. This common-random-
seed design usually reduces calibration noise and, more importantly, avoids
letting random pilot variation become another hidden experimental parameter.

Expected graph interface
------------------------
The calibration utilities use the graph attributes required by the chosen edge
model and calibration routine, principally graph.n, graph.m, graph.src, and
graph.dst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np


# ===========================================================================
# Basic validation / summaries
# ===========================================================================


def _edge_prob_array(graph, edge_prob=None) -> np.ndarray:
    """Return a validated float64 edge-probability/propensity vector."""
    m = int(graph.m)

    if edge_prob is None:
        prob = np.asarray(graph.prob, dtype=np.float64)
    else:
        prob = np.asarray(edge_prob, dtype=np.float64)

    if prob.shape != (m,):
        raise ValueError(f"edge_prob must have shape ({m},), got {prob.shape}.")
    if np.any(~np.isfinite(prob)):
        raise ValueError("edge probabilities / propensities must be finite.")
    if np.any(prob < 0.0):
        raise ValueError("edge probabilities / propensities must be nonnegative.")

    return prob


def _prior_array_simple(graph, prior_probs) -> np.ndarray:
    """
    Validate a prior vector for calibration utilities.

    This intentionally does not depend on the user's project-level
    _prior_array helper, so the module is self-contained.
    """
    p0 = np.asarray(prior_probs, dtype=np.float64)

    if p0.shape != (int(graph.n),):
        raise ValueError(
            f"prior_probs must have shape ({graph.n},), got {p0.shape}."
        )
    if np.any(~np.isfinite(p0)):
        raise ValueError("prior_probs must be finite.")
    if np.any((p0 < 0.0) | (p0 > 1.0)):
        raise ValueError("prior_probs must lie in [0, 1].")

    return p0


def _final_mean_from_result(result, n: Optional[int] = None) -> float:
    """Convert a pilot evaluator result to mean final activation."""
    if np.isscalar(result):
        value = float(result)
    elif isinstance(result, dict):
        if not result:
            raise ValueError("pilot_evaluator returned an empty dict.")
        arr = np.asarray(list(result.values()), dtype=np.float64)
        value = float(np.mean(arr))
    else:
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                "pilot_evaluator must return a scalar, a 1-D array, or a dict."
            )
        if n is not None and arr.shape != (int(n),):
            raise ValueError(
                f"pilot_evaluator returned shape {arr.shape}; expected ({n},)."
            )
        value = float(np.mean(arr))

    if not np.isfinite(value):
        raise ValueError("pilot_evaluator returned a non-finite mean activation.")

    return value


def secondary_activation_fraction(final_mean: float, prior_probs) -> float:
    """
    Compute

        g = (phi - phi0) / (1 - phi0),

    where phi is mean final activation and phi0 is mean initial activation.

    g=0 means no secondary diffusion beyond the prior.
    g=1 means all initially inactive probability mass becomes active.
    """
    p0 = np.asarray(prior_probs, dtype=np.float64)
    phi0 = float(np.mean(p0))

    if phi0 >= 1.0:
        raise ValueError(
            "secondary activation is undefined when mean initial activation is 1."
        )

    return (float(final_mean) - phi0) / (1.0 - phi0)


def edge_probability_summary(edge_prob) -> dict:
    """Cheap summary statistics worth storing for every calibrated instance."""
    p = np.asarray(edge_prob, dtype=np.float64)

    if p.size == 0:
        return {
            "edge_prob_mean": 0.0,
            "edge_prob_std": 0.0,
            "edge_prob_q50": 0.0,
            "edge_prob_q90": 0.0,
            "edge_prob_q99": 0.0,
            "edge_prob_max": 0.0,
            "edge_prob_frac_ge_0_1": 0.0,
            "edge_prob_frac_ge_0_25": 0.0,
            "edge_prob_frac_ge_0_5": 0.0,
        }

    q50, q90, q99 = np.quantile(p, [0.50, 0.90, 0.99])

    return {
        "edge_prob_mean": float(np.mean(p)),
        "edge_prob_std": float(np.std(p)),
        "edge_prob_q50": float(q50),
        "edge_prob_q90": float(q90),
        "edge_prob_q99": float(q99),
        "edge_prob_max": float(np.max(p)),
        "edge_prob_frac_ge_0_1": float(np.mean(p >= 0.10)),
        "edge_prob_frac_ge_0_25": float(np.mean(p >= 0.25)),
        "edge_prob_frac_ge_0_5": float(np.mean(p >= 0.50)),
    }


# ===========================================================================
# Edge-probability SHAPES
# ===========================================================================


def constant_base_probabilities(graph) -> np.ndarray:
    """
    Constant relative propensity w_e=1.

    Calibration gives p_e=min(max_prob, alpha).
    """
    return np.ones(int(graph.m), dtype=np.float64)


def weighted_cascade_base_probabilities(graph) -> np.ndarray:
    """
    Weighted-cascade relative propensity

        w_uv = 1 / indegree(v).

    Calibration gives

        p_uv(alpha) = min(max_prob, alpha / indegree(v)).
    """
    dst = np.asarray(graph.dst, dtype=np.int64)
    indeg = np.bincount(dst, minlength=int(graph.n)).astype(np.float64)

    base = np.zeros(int(graph.m), dtype=np.float64)
    denom = indeg[dst]
    mask = denom > 0.0
    base[mask] = 1.0 / denom[mask]
    return base


def heterogeneous_base_probabilities(
    graph,
    seed: int = 0,
    distribution: str = "uniform",
    dirichlet_concentration: float = 0.5,
) -> np.ndarray:
    """
    Draw ONE fixed heterogeneous relative-propensity pattern.

    Reuse the same realization for all calibration targets belonging to the
    same graph/edge-shape experiment.

    distributions
    -------------
    uniform
        Independent Uniform(0.1, 1.0) edge weights.

    beta
        Independent Beta(2, 5) edge weights.

    lognormal
        Independent exp(N(0,1)) edge weights, globally normalized so
        max(base)=1.

    dirichlet
        For every target node v independently, its incoming edge weights are

            (w_1v, ..., w_dv)
                ~ Dirichlet(kappa, ..., kappa),

        where kappa = dirichlet_concentration.

        Hence

            sum_{u -> v} w_uv = 1

        for every node with positive indegree.

        This makes the model directly comparable with weighted cascade,
        which corresponds to the deterministic equal-weight choice

            w_uv = 1 / indegree(v).
    """
    rng = np.random.default_rng(seed)

    m = int(graph.m)

    if distribution == "uniform":
        return rng.uniform(
            0.1,
            1.0,
            size=m,
        )

    if distribution == "beta":
        return rng.beta(
            2.0,
            5.0,
            size=m,
        )

    if distribution == "lognormal":
        x = rng.lognormal(
            mean=0.0,
            sigma=1.0,
            size=m,
        )

        if x.size and x.max() > 0:
            x /= x.max()

        return x

    if distribution == "dirichlet":
        kappa = float(
            dirichlet_concentration
        )

        if kappa <= 0.0:
            raise ValueError(
                "dirichlet_concentration must be > 0."
            )

        if m == 0:
            return np.empty(
                0,
                dtype=np.float64,
            )

        dst = np.asarray(
            graph.dst,
            dtype=np.int64,
        )

        # A Dirichlet can be generated by drawing independent
        # Gamma(kappa, 1) variables and normalizing them.
        x = rng.gamma(
            shape=kappa,
            scale=1.0,
            size=m,
        )

        incoming_sum = np.bincount(
            dst,
            weights=x,
            minlength=int(graph.n),
        )

        denom = incoming_sum[dst]

        base = np.zeros(
            m,
            dtype=np.float64,
        )

        mask = denom > 0.0

        base[mask] = (
            x[mask]
            / denom[mask]
        )

        return base

    raise ValueError(
        "distribution must be one of: "
        "'uniform', 'beta', 'lognormal', 'dirichlet'."
    )


def make_base_probabilities(
    graph,
    edge_model: str = "weighted_cascade",
    heterogeneous_seed: int = 0,
    dirichlet_concentration: float = 0.5,
) -> np.ndarray:
    """Convenience dispatcher for edge-probability shapes."""

    if edge_model == "constant":
        return constant_base_probabilities(
            graph
        )

    if edge_model == "weighted_cascade":
        return weighted_cascade_base_probabilities(
            graph
        )

    if edge_model == "heterogeneous_uniform":
        return heterogeneous_base_probabilities(
            graph,
            heterogeneous_seed,
            distribution="uniform",
        )

    if edge_model == "heterogeneous_beta":
        return heterogeneous_base_probabilities(
            graph,
            heterogeneous_seed,
            distribution="beta",
        )

    if edge_model == "heterogeneous_lognormal":
        return heterogeneous_base_probabilities(
            graph,
            heterogeneous_seed,
            distribution="lognormal",
        )

    if edge_model == "heterogeneous_dirichlet":
        return heterogeneous_base_probabilities(
            graph,
            heterogeneous_seed,
            distribution="dirichlet",
            dirichlet_concentration=(
                dirichlet_concentration
            ),
        )

    raise ValueError(
        "edge_model must be one of: "
        "'constant', "
        "'weighted_cascade', "
        "'heterogeneous_uniform', "
        "'heterogeneous_beta', "
        "'heterogeneous_lognormal', "
        "'heterogeneous_dirichlet'."
    )


def scaled_edge_probabilities(
    base_prob,
    alpha: float,
    max_prob: float = 1.0,
) -> np.ndarray:
    """Return p_e(alpha)=min(max_prob, alpha*w_e)."""
    alpha = float(alpha)
    max_prob = float(max_prob)

    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative.")
    if not (0.0 < max_prob <= 1.0):
        raise ValueError("max_prob must lie in (0,1].")

    base = np.asarray(base_prob, dtype=np.float64)
    if np.any(base < 0.0):
        raise ValueError("base_prob must be nonnegative.")

    return np.minimum(max_prob, alpha * base)


# ===========================================================================
# Optional seed-prior helper
# ===========================================================================


def make_random_prior(
    graph,
    support_fraction: float,
    seed: int = 0,
    value_model: str = "deterministic",
    low: float = 0.0,
    high: float = 1.0,
    exact_support: bool = True,
) -> np.ndarray:
    """
    Construct a reproducible random seed prior.

    support_fraction is the fraction of nodes with nonzero prior probability.

    value_model
    -----------
    deterministic : selected nodes have prior 1
    uniform       : selected nodes have U(low, high) prior probability
    constant      : selected nodes have prior=high

    If exact_support=True, exactly round(support_fraction*n) nodes are selected.
    Otherwise each node is independently included with support_fraction.
    """
    s = float(support_fraction)
    n = int(graph.n)

    if not (0.0 <= s <= 1.0):
        raise ValueError("support_fraction must lie in [0,1].")
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError("Require 0 <= low <= high <= 1.")

    rng = np.random.default_rng(seed)
    prior = np.zeros(n, dtype=np.float64)

    if exact_support:
        k = int(round(s * n))
        k = min(max(k, 0), n)
        support = rng.choice(n, size=k, replace=False) if k else np.empty(0, int)
    else:
        support = np.flatnonzero(rng.random(n) < s)

    if value_model == "deterministic":
        prior[support] = 1.0
    elif value_model == "uniform":
        prior[support] = rng.uniform(low, high, size=len(support))
    elif value_model == "constant":
        prior[support] = high
    else:
        raise ValueError(
            "value_model must be 'deterministic', 'uniform', or 'constant'."
        )

    return prior


# ===========================================================================
# PRIMARY calibration: seed-conditioned secondary activation
# ===========================================================================


@dataclass
class CalibrationPoint:
    alpha: float
    mean_final_activation: float
    secondary_activation: float
    repeat_std: float
    clipped_fraction: float



def _evaluate_alpha(
    graph,
    base_prob: np.ndarray,
    alpha: float,
    prior_probs: np.ndarray,
    pilot_evaluator: Callable,
    pilot_num_sim: int,
    pilot_seeds: Iterable[int],
    max_prob: float,
) -> CalibrationPoint:
    """Evaluate one candidate alpha using repeated pilot estimates."""
    edge_prob = scaled_edge_probabilities(base_prob, alpha, max_prob=max_prob)

    means = []
    for seed in pilot_seeds:
        result = pilot_evaluator(edge_prob, int(pilot_num_sim), int(seed))
        means.append(_final_mean_from_result(result, n=int(graph.n)))

    mean_final = float(np.mean(means))
    repeat_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0

    g = secondary_activation_fraction(mean_final, prior_probs)

    return CalibrationPoint(
        alpha=float(alpha),
        mean_final_activation=mean_final,
        secondary_activation=float(g),
        repeat_std=repeat_std,
        clipped_fraction=float(np.mean(edge_prob >= max_prob - 1e-14)),
    )


def calibrate_edge_probabilities_to_secondary_activation(
    graph,
    base_prob,
    prior_probs,
    target_secondary_activation: float,
    pilot_evaluator: Callable,
    *,
    pilot_num_sim: int = 250,
    pilot_repeats: int = 1,
    pilot_seed: int = 12345,
    max_prob: float = 1.0,
    initial_alpha: float = 1.0,
    alpha_growth: float = 2.0,
    max_alpha: float = 1e8,
    binary_search_steps: int = 10,
    target_tol: float = 0.02,
    min_binary_steps: int = 3,
    return_trace: bool = True,
):
    """
    Calibrate scalar edge strength CONDITIONAL ON the chosen seed prior.

    Parameters
    ----------
    graph : ICGraph-like
    base_prob : array-like, shape (m,)
        Fixed edge-probability SHAPE w_e.
    prior_probs : array-like, shape (n,)
        Fixed seed prior for this experimental instance.
    target_secondary_activation : float
        Desired

            g = (mean(final) - mean(prior)) / (1 - mean(prior)).

        Typical exploratory targets might be 0.10, 0.35, 0.60.
    pilot_evaluator : callable
        pilot_evaluator(edge_prob, num_sim, seed) -> scalar/array/dict.
        It should estimate final IC activation under the FIXED prior_probs that
        this closure/wrapper captures.
    pilot_num_sim : int
        Cheap MC simulations per pilot call.
    pilot_repeats : int
        Independent pilot calls averaged at each candidate alpha.
    pilot_seed : int
        Base seed. The same set of pilot seeds is reused at every alpha.
    max_prob : float
        Edge-probability cap, normally 1.
    initial_alpha : float
        Initial positive alpha used to bracket the target.
    alpha_growth : float
        Multiplicative bracket expansion factor.
    max_alpha : float
        Safety cap for alpha.
    binary_search_steps : int
        Maximum bisection steps after bracketing.
    target_tol : float
        Stop when |estimated_g - target_g| <= target_tol, after at least
        min_binary_steps.
    min_binary_steps : int
        Avoid accepting an accidentally lucky noisy pilot too early.
    return_trace : bool
        If True, include every evaluated calibration point in info['trace'].

    Returns
    -------
    calibrated_prob : np.ndarray
        Final p_e=min(max_prob, alpha*w_e).
    info : dict
        Calibration metadata.

    Notes
    -----
    Expected IC spread is monotone in every edge probability. Therefore the
    EXPECTED secondary activation is monotone in alpha. Pilot MC estimates are
    noisy, so the function keeps the best observed point rather than blindly
    returning the final binary-search midpoint.
    """
    target = float(target_secondary_activation)

    if not (0.0 <= target <= 1.0):
        raise ValueError("target_secondary_activation must lie in [0,1].")
    if pilot_num_sim <= 0:
        raise ValueError("pilot_num_sim must be positive.")
    if pilot_repeats <= 0:
        raise ValueError("pilot_repeats must be positive.")
    if initial_alpha <= 0.0:
        raise ValueError("initial_alpha must be positive.")
    if alpha_growth <= 1.0:
        raise ValueError("alpha_growth must be >1.")
    if max_alpha <= 0.0:
        raise ValueError("max_alpha must be positive.")

    base = _edge_prob_array(graph, base_prob)
    p0 = _prior_array_simple(graph, prior_probs)

    if float(np.mean(p0)) >= 1.0:
        raise ValueError("Calibration is undefined when mean prior activation is 1.")

    if not np.any(base > 0.0):
        if target == 0.0:
            zero = np.zeros_like(base)
            return zero, {
                "feasible": True,
                "target_secondary_activation": 0.0,
                "achieved_secondary_activation": 0.0,
                "alpha": 0.0,
                "mean_initial_activation": float(np.mean(p0)),
                "mean_final_activation": float(np.mean(p0)),
                "pilot_num_sim": int(pilot_num_sim),
                "pilot_repeats": int(pilot_repeats),
                "clipped_fraction": 0.0,
                "trace": [],
            }
        raise ValueError("Positive target cannot be reached from all-zero base_prob.")

    pilot_seeds = [int(pilot_seed) + i for i in range(int(pilot_repeats))]
    trace: list[CalibrationPoint] = []

    # alpha=0 has expected g=0 exactly, so no MC is needed there.
    low_alpha = 0.0
    low_g = 0.0

    if target == 0.0:
        zero = np.zeros_like(base)
        return zero, {
            "feasible": True,
            "target_secondary_activation": 0.0,
            "achieved_secondary_activation": 0.0,
            "alpha": 0.0,
            "mean_initial_activation": float(np.mean(p0)),
            "mean_final_activation": float(np.mean(p0)),
            "pilot_num_sim": int(pilot_num_sim),
            "pilot_repeats": int(pilot_repeats),
            "clipped_fraction": 0.0,
            "trace": [],
            **edge_probability_summary(zero),
        }

    # ------------------------------------------------------------------
    # Bracket target by increasing alpha.
    # ------------------------------------------------------------------
    high_alpha = float(initial_alpha)
    high_point = None

    while True:
        high_point = _evaluate_alpha(
            graph,
            base,
            high_alpha,
            p0,
            pilot_evaluator,
            pilot_num_sim,
            pilot_seeds,
            max_prob,
        )
        trace.append(high_point)

        if high_point.secondary_activation >= target:
            break

        low_alpha = high_alpha
        low_g = high_point.secondary_activation

        # If every positive edge is already saturated, larger alpha cannot
        # change the experimental instance at all.
        edge_high = scaled_edge_probabilities(base, high_alpha, max_prob=max_prob)
        positive = base > 0.0
        fully_saturated = bool(
            np.all(edge_high[positive] >= max_prob - 1e-14)
        )

        if fully_saturated:
            best = max(trace, key=lambda q: q.secondary_activation)
            raise ValueError(
                f"target_secondary_activation={target:.6g} is infeasible; "
                f"with all positive base edges saturated at {max_prob}, "
                f"pilot secondary activation is only "
                f"g≈{best.secondary_activation:.6g}."
            )

        next_alpha = high_alpha * alpha_growth
        if next_alpha > max_alpha:
            raise ValueError(
                f"Failed to bracket target g={target:.6g} before alpha exceeded "
                f"max_alpha={max_alpha:.6g}. Last pilot g≈"
                f"{high_point.secondary_activation:.6g}."
            )

        high_alpha = next_alpha

    # Best point seen so far. Include the analytic alpha=0 point only for the
    # distance comparison, not as an MC trace entry.
    best_point = min(
        trace,
        key=lambda q: abs(q.secondary_activation - target),
    )
    if abs(low_g - target) < abs(best_point.secondary_activation - target):
        # This branch only matters when the previous low point was evaluated.
        low_candidates = [q for q in trace if q.alpha == low_alpha]
        if low_candidates:
            best_point = low_candidates[-1]

    # ------------------------------------------------------------------
    # Noisy binary search inside bracket.
    # ------------------------------------------------------------------
    for step in range(int(binary_search_steps)):
        mid_alpha = 0.5 * (low_alpha + high_alpha)

        point = _evaluate_alpha(
            graph,
            base,
            mid_alpha,
            p0,
            pilot_evaluator,
            pilot_num_sim,
            pilot_seeds,
            max_prob,
        )
        trace.append(point)

        if abs(point.secondary_activation - target) < abs(
            best_point.secondary_activation - target
        ):
            best_point = point

        if point.secondary_activation < target:
            low_alpha = mid_alpha
            low_g = point.secondary_activation
        else:
            high_alpha = mid_alpha
            high_point = point

        if (
            step + 1 >= int(min_binary_steps)
            and abs(best_point.secondary_activation - target) <= target_tol
        ):
            break

    calibrated = scaled_edge_probabilities(
        base,
        best_point.alpha,
        max_prob=max_prob,
    )

    info = {
        "feasible": True,
        "target_secondary_activation": target,
        "achieved_secondary_activation": best_point.secondary_activation,
        "secondary_activation_abs_error": abs(
            best_point.secondary_activation - target
        ),
        "alpha": best_point.alpha,
        "mean_initial_activation": float(np.mean(p0)),
        "seed_support_fraction": float(np.mean(p0 > 0.0)),
        "mean_final_activation": best_point.mean_final_activation,
        "pilot_repeat_std_final_mean": best_point.repeat_std,
        "pilot_num_sim": int(pilot_num_sim),
        "pilot_repeats": int(pilot_repeats),
        "pilot_seed": int(pilot_seed),
        "clipped_fraction": best_point.clipped_fraction,
        **edge_probability_summary(calibrated),
    }

    if return_trace:
        info["trace"] = [
            {
                "alpha": q.alpha,
                "mean_final_activation": q.mean_final_activation,
                "secondary_activation": q.secondary_activation,
                "repeat_std": q.repeat_std,
                "clipped_fraction": q.clipped_fraction,
            }
            for q in trace
        ]

    return calibrated, info



def calibrate_secondary_activation_grid(
    graph,
    base_prob,
    prior_probs,
    targets,
    pilot_evaluator: Callable,
    **kwargs,
) -> dict:
    """
    Calibrate several secondary-activation regimes for ONE fixed seed prior.

    The edge-probability shape and seed prior stay fixed; only alpha changes.
    """
    result = {}

    # Increasing order improves interpretability of returned records. The
    # calibrator itself is independent per target, intentionally avoiding
    # hidden warm-start state between experimental conditions.
    for target in sorted(float(x) for x in targets):
        prob, info = calibrate_edge_probabilities_to_secondary_activation(
            graph,
            base_prob,
            prior_probs,
            target,
            pilot_evaluator,
            **kwargs,
        )
        result[target] = {"prob": prob, "info": info}

    return result


# ===========================================================================
# Convenience example wrapper
# ===========================================================================


def calibration_demo(
    graph,
    prior_probs,
    pilot_evaluator: Callable,
    *,
    edge_model: str = "weighted_cascade",
    heterogeneous_seed: int = 0,
    targets=(0.10, 0.35, 0.60),
    **calibration_kwargs,
):
    """
    Convenience demonstration of the intended pipeline:

        graph -> edge shape -> fixed seed prior -> calibrated regimes.
    """
    base = make_base_probabilities(
        graph,
        edge_model=edge_model,
        heterogeneous_seed=heterogeneous_seed,
    )

    return calibrate_secondary_activation_grid(
        graph,
        base,
        prior_probs,
        targets,
        pilot_evaluator,
        **calibration_kwargs,
    )


__all__ = [
    # Primary pipeline
    "secondary_activation_fraction",
    "calibrate_edge_probabilities_to_secondary_activation",
    "calibrate_secondary_activation_grid",
    "calibration_demo",
    # Edge shapes
    "constant_base_probabilities",
    "weighted_cascade_base_probabilities",
    "heterogeneous_base_probabilities",
    "make_base_probabilities",
    "scaled_edge_probabilities",
    # Seed helper
    "make_random_prior",
    # Summaries
    "edge_probability_summary",
]
