import numpy as np
import pandas as pd
import time
import gzip
import re
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from edge_probabilities import generate_edge_probabilities
from data.synthetic_graph_generation import (
    forest_fire_graph,
    er_graph,
    ba_graph,
    watts_strogatz_graph,
    equal_sbm_parameters,
    sbm_graph,
)
from GraphIC import (
    ICGraph,
    optimized_independent_cascade,
    swe_no,
    swe_no_cavity,
    additive_swe, 
    additive_swe_cavity,
    swe,
    swe_cavity,
    swe_hib_cavity,
    Naive,
    dmp_est_r2,
    dmp_est,
    dmp_inf,
    modified_ALE,
    cavity_ALE,
    modified_ALE_cavity,
    ALE_heuristic,
    SPM,
    SP1M,
    pagerank,
    sss,
    sss_noself,
    mia,
    get_best_parameters,
)

def evaluate_predictions(preds, true_probs, percentile=95):
    preds = np.asarray(preds)
    true_probs = np.asarray(true_probs)

    diff = preds - true_probs

    rmse = np.sqrt(np.mean(diff * diff))
    mae = np.mean(np.abs(diff))

    pearson = np.corrcoef(preds, true_probs)[0, 1]

    spearman = spearmanr(
        preds,
        true_probs,
    ).statistic

    threshold = np.percentile(true_probs, percentile)
    y_binary = (true_probs >= threshold).astype(np.uint8)

    # roc_auc_score fails if only one class exists.
    if y_binary.min() == y_binary.max():
        auc = np.nan
    else:
        auc = roc_auc_score(y_binary, preds)

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Pearson": float(pearson),
        "Spearman": float(spearman),
        "AUC": float(auc),
    }

def run_method(
    method,
    graph,
    prior_probs,
    true_probs,
    *,
    max_t=10,
    max_layers=1,
    theta=0.01,
):
    """
    Returns
    -------
    preds : ndarray
    params : dict
    tuning_time : float
    runtime : float
    """

    params = {}
    tuning_time = 0.0

    # ----------------------------------------------------------
    # swe_no / swe_no_cavity: tune T and layers
    # ----------------------------------------------------------

    if method == "dmp_est_r2":
        graph.ensure_r2_triangles()
        
    if method in {"swe_no", "swe_no_cavity"}:

        start = time.perf_counter()

        best_T, best_l = get_best_parameters(
            graph,
            prior_probs,
            true_probs,
            method=method,
            max_t=max_t,
            max_layers=max_layers,
        )

        tuning_time = time.perf_counter() - start

        fn = swe_no if method == "swe_no" else swe_no_cavity

        start = time.perf_counter()

        preds = fn(
            graph,
            prior_probs,
            best_T,
            a=1.0,
            layers=best_l,
        )

        runtime = time.perf_counter() - start

        params = {
            "T": best_T,
            "layers": best_l,
        }

        return preds, params, tuning_time, runtime

    # ----------------------------------------------------------
    # Methods requiring only T tuning
    # ----------------------------------------------------------

    tuned_methods = {
        "swe":
            lambda T: swe(graph, prior_probs, T),

        "swe_cavity":
            lambda T: swe_cavity(graph, prior_probs, T),

        "additive_swe":
            lambda T: additive_swe(graph, prior_probs, T),

        "additive_swe_cavity":
            lambda T: additive_swe_cavity(graph, prior_probs, T),

        "swe_hib_cavity":
            lambda T: swe_hib_cavity(
                graph, prior_probs, T, eps=1e-20
            ),

        "Naive":
            lambda T: Naive(
                graph, prior_probs, T, eps=1e-12
            ),

        "dmp_est_r2":
            lambda T: dmp_est_r2(
                graph, prior_probs, T
            ),

        "dmp_est":
            lambda T: dmp_est(
                graph, prior_probs, T
            ),

        "modified_ALE":
            lambda T: modified_ALE(
                graph, prior_probs, T
            ),

        "ALE_heuristic":
            lambda T: ALE_heuristic(
                graph, prior_probs, T
            ),

        "cavity_ALE": 
            lambda T: cavity_ALE(
                graph, prior_probs, T
            ),

        "modified_ALE_cavity":
            lambda T: modified_ALE_cavity(
                graph, prior_probs, T
            ),
    }

    if method in tuned_methods:

        start = time.perf_counter()

        best_T = get_best_parameters(
            graph,
            prior_probs,
            true_probs,
            method=method,
            max_t=max_t,
        )

        tuning_time = time.perf_counter() - start

        start = time.perf_counter()

        preds = tuned_methods[method](best_T)

        runtime = time.perf_counter() - start

        params = {"T": best_T}

        return preds, params, tuning_time, runtime

    # ----------------------------------------------------------
    # Methods without T tuning
    # ----------------------------------------------------------

    fixed_methods = {
        "dmp_inf":
            lambda: dmp_inf(
                graph,
                prior_probs,
                eps=1e-20,
                max_iter=100,
            ),

        "SPM":
            lambda: SPM(
                graph,
                prior_probs,
            ),

        "SP1M":
            lambda: SP1M(
                graph,
                prior_probs,
            ),

        "pagerank":
            lambda: pagerank(
                graph,
                prior_probs=prior_probs,
                alpha=0.85,
                max_iter=100,
                tol=1e-12,
                transition="edge_probs",
                eps=1e-20,
            ),

        "priorprobs":
            lambda: np.asarray(prior_probs),

        "sss":
            lambda: sss(
                graph,
                prior_probs,
                tol=1e-8,
                max_iter=1000,
                eps=1e-12,
            ),

        "sss_noself":
            lambda: sss_noself(
                graph,
                prior_probs,
                tol=1e-8,
                max_iter=1000,
                eps=1e-12,
            ),

        "mia":
            lambda: mia(
                graph,
                prior_probs,
                theta=theta,
                eps=1e-15,
            ),

    }

    if method not in fixed_methods:
        raise ValueError(f"Unknown method: {method}")

    start = time.perf_counter()
    preds = fixed_methods[method]()
    runtime = time.perf_counter() - start

    return preds, params, 0.0, runtime

def load_directed_graph(
    path,
    *,
    comments="#",
    one_based=None,
    prob_dtype=np.float32,
):
    """
    Load a directed edge-list file directly into ICGraph.

    Each line represents exactly one directed edge:
        u v   means   u -> v

    Parameters
    ----------
    path : str
        Edge-list path.

    comments : str or None
        Lines beginning with this string are ignored.

    one_based : bool or None
        True:
            IDs are assumed to be 1, ..., n and converted to 0, ..., n-1.

        False:
            IDs are assumed already zero-based.

        None:
            Arbitrary node labels are remapped to contiguous IDs
            0, ..., n-1.

            This is the safest option for datasets such as Wiki-Vote,
            where node IDs need not form a contiguous range.

    prob_dtype : numpy dtype
        Storage type for edge probabilities.

    Returns
    -------
    ICGraph
    """

    src_raw = []
    dst_raw = []
    
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if comments is not None and line.startswith(comments):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            src_raw.append(u)
            dst_raw.append(v)

    if len(src_raw) == 0:
        return ICGraph.from_edges(
            0,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=prob_dtype),
            prob_dtype=prob_dtype,
        )

    src_raw = np.asarray(src_raw, dtype=np.int64)
    dst_raw = np.asarray(dst_raw, dtype=np.int64)

    # ----------------------------------------------------------
    # Arbitrary labels -> contiguous 0,...,n-1 IDs
    # ----------------------------------------------------------

    if one_based is None:

        nodes = np.unique(
            np.concatenate((src_raw, dst_raw))
        )

        src = np.searchsorted(nodes, src_raw)
        dst = np.searchsorted(nodes, dst_raw)

        n = len(nodes)

        labels = nodes

    # ----------------------------------------------------------
    # Known 1-based indexing
    # ----------------------------------------------------------

    elif one_based:

        src = src_raw - 1
        dst = dst_raw - 1

        if src.min() < 0 or dst.min() < 0:
            raise ValueError(
                "Negative node ID after one-based conversion."
            )

        n = int(
            max(src.max(), dst.max()) + 1
        )

        labels = None

    # ----------------------------------------------------------
    # Known 0-based indexing
    # ----------------------------------------------------------

    else:

        if src_raw.min() < 0 or dst_raw.min() < 0:
            raise ValueError(
                "Negative node IDs are not supported."
            )

        src = src_raw
        dst = dst_raw

        n = int(
            max(src.max(), dst.max()) + 1
        )

        labels = None

    node_dtype = (
        np.int32
        if n <= np.iinfo(np.int32).max
        else np.int64
    )

    src = src.astype(node_dtype, copy=False)
    dst = dst.astype(node_dtype, copy=False)

    probs = np.ones(
        len(src),
        dtype=prob_dtype,
    )

    # ----------------------------------------------------------
    # Remove self-loops
    # ----------------------------------------------------------
    
    non_self = src != dst
    
    num_self_loops = int((~non_self).sum())
    
    if num_self_loops:
        print(f"Removed {num_self_loops:,} self-loops.")
    
    src = src[non_self]
    dst = dst[non_self]
    
    
    # ----------------------------------------------------------
    # Remove duplicate directed edges
    # ----------------------------------------------------------
    
    keys = (
        src.astype(np.int64, copy=False) * np.int64(n)
        + dst.astype(np.int64, copy=False)
    )
    
    _, unique_idx = np.unique(
        keys,
        return_index=True,
    )
    
    num_duplicates = len(src) - len(unique_idx)
    
    if num_duplicates:
        print(f"Removed {num_duplicates:,} duplicate directed edges.")
    
    src = src[unique_idx]
    dst = dst[unique_idx]
    
    
    # ----------------------------------------------------------
    # NOW create probabilities
    # ----------------------------------------------------------
    
    probs = np.ones(
        len(src),
        dtype=prob_dtype,
    )
    
    print(
        "DEBUG lengths:",
        len(src),
        len(dst),
        len(probs),
    )
    
    return ICGraph.from_edges(
        n,
        src,
        dst,
        probs,
        prob_dtype=prob_dtype,
        labels=labels,
    )

def load_bidirectional_graph(
    path,
    *,
    comments="#",
    one_based=None,
    plain_header=False,   
    prob_dtype=np.float32,
):
    """
    Load an undirected edge list into ICGraph, storing every undirected
    edge {u,v} as the two directed edges u->v and v->u.

    Handles:
      - ordinary text files
      - .gz files
      - blank lines
      - comment lines
      - SNAP headers such as:
            # Nodes: 7115 Edges: 103689
      - optional plain first-line header:
            7115 103689
      - files with no n,m header
      - arbitrary/non-contiguous node IDs

    Parameters
    ----------
    path : str
        Path to edge-list file.

    comments : str or None
        Comment prefix.

    one_based : bool or None
        True:
            Interpret IDs as 1,...,n and subtract 1.

        False:
            Interpret IDs as already zero-based.

        None:
            Remap arbitrary labels to contiguous IDs 0,...,n-1.
            This is the safest default.

    prob_dtype : numpy dtype
        Type used for edge probabilities.

    Returns
    -------
    ICGraph
    """

    opener = gzip.open if str(path).endswith(".gz") else open

    src_raw = []
    dst_raw = []

    header_n = None
    header_m = None
    first_data_line = True

    # Matches e.g.
    #   # Nodes: 7115 Edges: 103689
    snap_header = re.compile(
        r"Nodes:\s*(\d+).*Edges:\s*(\d+)",
        re.IGNORECASE,
    )

    with opener(path, "rt") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            # --------------------------------------------------
            # Comment/header line
            # --------------------------------------------------
            if comments is not None and line.startswith(comments):

                match = snap_header.search(line)

                if match:
                    header_n = int(match.group(1))
                    header_m = int(match.group(2))

                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            # --------------------------------------------------
            # Optional plain "n m" header
            # --------------------------------------------------
            if first_data_line:
                first_data_line = False
    
                # Explicitly requested numeric n,m header.
                # Example:
                #     15229 31376
                if plain_header:
                    try:
                        header_n = int(parts[0])
                        header_m = int(parts[1])
                        continue
                    except ValueError:
                        raise ValueError(
                            "plain_header=True, but the first data line "
                            "does not contain two integers."
                        )
    
                # Unusual textual header.
                if len(parts) >= 3 and parts[0].lower() in {
                    "nodes",
                    "vertices",
                }:
                    continue
            
            # --------------------------------------------------
            # Normal edge
            # --------------------------------------------------
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                # Textual column header such as:
                # FromNodeId ToNodeId
                continue

            src_raw.append(u)
            dst_raw.append(v)

    if len(src_raw) == 0:
        n = header_n if header_n is not None else 0

        return ICGraph.from_edges(
            n,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=prob_dtype),
            prob_dtype=prob_dtype,
        )

    src_raw = np.asarray(src_raw, dtype=np.int64)
    dst_raw = np.asarray(dst_raw, dtype=np.int64)

    # ==========================================================
    # Node indexing
    # ==========================================================

    if one_based is None:
        # Safest option:
        # arbitrary original labels -> 0,...,n-1

        labels = np.unique(
            np.concatenate((src_raw, dst_raw))
        )

        src = np.searchsorted(labels, src_raw)
        dst = np.searchsorted(labels, dst_raw)

        n = len(labels)

        # Warn only conceptually via validation:
        # header_n may include isolated vertices absent from edge list.
        # We cannot reconstruct their original labels automatically.

    elif one_based:
        src = src_raw - 1
        dst = dst_raw - 1

        if src.min() < 0 or dst.min() < 0:
            raise ValueError(
                "Negative node ID after one-based conversion. "
                "The input is probably not one-based."
            )

        if header_n is not None:
            n = header_n
        else:
            n = int(max(src.max(), dst.max()) + 1)

        labels = None

    else:
        src = src_raw
        dst = dst_raw

        if src.min() < 0 or dst.min() < 0:
            raise ValueError(
                "Negative node IDs are not supported."
            )

        if header_n is not None:
            n = header_n
        else:
            n = int(max(src.max(), dst.max()) + 1)

        labels = None

    # ----------------------------------------------------------
    # Remove self-loops BEFORE bidirectional expansion
    # ----------------------------------------------------------
    
    non_self = src != dst
    num_self_loops = int((~non_self).sum())
    
    if num_self_loops:
        print(f"Removed {num_self_loops:,} self-loops.")
    
    src = src[non_self]
    dst = dst[non_self]
    
    
    # ----------------------------------------------------------
    # Canonicalize undirected edges
    # ----------------------------------------------------------
    
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    
    pairs = np.column_stack((u, v))
    
    original_m = len(pairs)
    
    pairs = np.unique(pairs, axis=0)
    
    duplicates_removed = original_m - len(pairs)
    
    if duplicates_removed:
        print(
            f"Removed {duplicates_removed:,} duplicate "
            f"undirected edge entries."
        )
    
    src = pairs[:, 0]
    dst = pairs[:, 1]
    
    
    # ----------------------------------------------------------
    # Expand to two directed edges
    # ----------------------------------------------------------
    
    node_dtype = (
        np.int32
        if n <= np.iinfo(np.int32).max
        else np.int64
    )
    
    src = src.astype(node_dtype, copy=False)
    dst = dst.astype(node_dtype, copy=False)
    
    m_undirected = len(src)
    
    src_dir = np.empty(
        2 * m_undirected,
        dtype=node_dtype,
    )
    
    dst_dir = np.empty(
        2 * m_undirected,
        dtype=node_dtype,
    )
    
    src_dir[:m_undirected] = src
    dst_dir[:m_undirected] = dst
    
    src_dir[m_undirected:] = dst
    dst_dir[m_undirected:] = src
    
    
    # ----------------------------------------------------------
    # Probabilities MUST match directed edge arrays
    # ----------------------------------------------------------
    
    probs = np.ones(
        len(src_dir),
        dtype=prob_dtype,
    )
    
    print(
        "DEBUG lengths:",
        len(src_dir),
        len(dst_dir),
        len(probs),
    )
    
    return ICGraph.from_edges(
        n,
        src_dir,
        dst_dir,
        probs,
        prob_dtype=prob_dtype,
        labels=labels,
    )

def make_bidirectional(
    graph,
    *,
    prob_dtype=np.float32,
):
    """
    Convert an ICGraph into a bidirectional graph.

    For every directed edge u -> v, ensure that both
    u -> v and v -> u exist.

    Duplicate edges are removed.
    Self-loops are removed.
    """

    n = graph.n

    src = graph.src
    dst = graph.dst

    # Remove self-loops, if any.
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # Canonical undirected representation.
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)

    keys = (
        u.astype(np.int64, copy=False) * np.int64(n)
        + v.astype(np.int64, copy=False)
    )

    _, idx = np.unique(
        keys,
        return_index=True,
    )

    u = u[idx]
    v = v[idx]

    m = len(u)

    node_dtype = graph.src.dtype

    src_bi = np.empty(2 * m, dtype=node_dtype)
    dst_bi = np.empty(2 * m, dtype=node_dtype)

    src_bi[:m] = u
    dst_bi[:m] = v

    src_bi[m:] = v
    dst_bi[m:] = u

    probs = np.ones(
        2 * m,
        dtype=prob_dtype,
    )

    return ICGraph.from_edges(
        n,
        src_bi,
        dst_bi,
        probs,
        prob_dtype=prob_dtype,
    )

def load_dataset(
    dataset, 
    bidirectional = False,
    n=10000,
    p=0.3,
    r=0.3,
    seed=42,
    m=10,
    k=10,
    beta=0.1,
    num_blocks=10,
    avg_degree=10,
    ratio=10,
):
    if dataset == "NetHept":
        graph = load_bidirectional_graph(
            "data/NetHEHT.txt",
            one_based=False,
            plain_header=True, 
            prob_dtype=np.float32,
        )
    elif dataset == "WikiVote":
        graph = load_directed_graph(
            "data/wiki-Vote.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "Enron":
        graph = load_bidirectional_graph(
            "data/email-Enron.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "Epinions":
        graph = load_directed_graph(
            "data/soc-Epinions1.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "Slashdot":
        graph = load_directed_graph(
            "data/soc-Slashdot0902.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "CondMat":
        graph = load_bidirectional_graph(
            "data/ca-CondMat.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "citHepTh":
        graph = load_directed_graph(
            "data/cit-HepTh.txt.gz",
            comments="#",
            one_based=None,
            prob_dtype=np.float32,
        )
    elif dataset == "ForestFire":
        graph = forest_fire_graph(
            n,
            p=p,
            r=r
        )
    elif dataset == "ErdosRenyi":
        graph = er_graph(
            n=n,
            p=m / (n - 1),
            seed=seed,
        )
    elif dataset == "BarabasiAlbert":
        graph = ba_graph(
            n=n,
            m=m,
            seed=seed,
        )
    elif dataset == "WattsStrogatz":
        graph = watts_strogatz_graph(
            n=n,
            k=k,
            beta=beta,
            seed=seed,
        )
    elif dataset == "StochasticBlockModel":
        sizes, P = equal_sbm_parameters(
            n=n,
            num_blocks=num_blocks,
            avg_degree=avg_degree,
            ratio=ratio,
        )
        graph = sbm_graph(
            sizes,
            P,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if bidirectional:
        graph = make_bidirectional(graph)

    return graph