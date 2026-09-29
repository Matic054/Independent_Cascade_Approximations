import numpy as np

def generate_edge_probabilities(
    graph,
    method="weighted_cascade",
    *,
    value=0.01,
    low=0.0,
    high=0.1,
    alpha=1.0,
    beta_a=2.0,
    beta_b=18.0,
    choices=(0.001, 0.01, 0.1),
    seed=None,
):
    """
    Generate IC edge probabilities aligned with graph.src / graph.dst.

    Methods
    -------
    weighted_cascade
        p_uv = 1 / indegree(v)

    scaled_weighted_cascade
        p_uv = min(1, alpha / indegree(v))

    constant
        p_uv = value

    uniform
        p_uv ~ Uniform(low, high)

    beta
        p_uv ~ Beta(beta_a, beta_b)

    trivalency
        p_uv chosen uniformly from `choices`
    """

    rng = np.random.default_rng(seed)
    dtype = graph.prob.dtype

    if method == "weighted_cascade":

        indegree = np.bincount(
            graph.dst,
            minlength=graph.n,
        )

        p = 1.0 / indegree[graph.dst]

        return p.astype(dtype)


    elif method == "scaled_weighted_cascade":

        indegree = np.bincount(
            graph.dst,
            minlength=graph.n,
        )

        p = alpha / indegree[graph.dst]

        p = np.minimum(p, 1.0)

        return p.astype(dtype)


    elif method == "constant":

        if not (0.0 <= value <= 1.0):
            raise ValueError("value must lie in [0,1].")

        return np.full(
            graph.m,
            value,
            dtype=dtype,
        )


    elif method == "uniform":

        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                "Require 0 <= low <= high <= 1."
            )

        return rng.uniform(
            low,
            high,
            size=graph.m,
        ).astype(dtype)


    elif method == "beta":

        if beta_a <= 0 or beta_b <= 0:
            raise ValueError(
                "Beta parameters must be positive."
            )

        return rng.beta(
            beta_a,
            beta_b,
            size=graph.m,
        ).astype(dtype)


    elif method == "trivalency":

        choices = np.asarray(
            choices,
            dtype=dtype,
        )

        if np.any((choices < 0) | (choices > 1)):
            raise ValueError(
                "All choices must lie in [0,1]."
            )

        return rng.choice(
            choices,
            size=graph.m,
            replace=True,
        ).astype(dtype)


    else:
        raise ValueError(
            f"Unknown probability method: {method}"
        )