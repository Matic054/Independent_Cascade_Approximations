#!/usr/bin/env python3
"""
Create a LaTeX table of structural statistics for the seven real-world
IC benchmark networks.

Assumes `utilities.load_dataset(name)` is available and returns, or
contains, an ICGraph-like object with attributes `n`, `src`, and `dst`.

Definitions
-----------
Edges:
    Directed datasets -> distinct directed arcs.
    Undirected datasets -> distinct undirected edges.

Average degree:
    Computed on the simple undirected projection for every dataset.

Average clustering coefficient:
    Mean local clustering coefficient on the simple undirected projection.

Triangles:
    Number of distinct triangles in the simple undirected projection.

Reciprocity:
    Optional extra row, useful for feedback/cavity methods. For directed
    datasets this is the fraction of directed arcs whose reverse arc exists.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

from utilities import load_dataset


DATASETS = (
    "NetHept",
    "WikiVote",
    "Enron",
    "Epinions",
    "Slashdot",
    "CondMat",
    "citHepTh",
)

# Semantic direction of the source datasets. Do not infer this from the
# internal ICGraph representation because undirected datasets are commonly
# represented using two opposite arcs per undirected edge.
IS_DIRECTED = {
    "NetHept": False,
    "WikiVote": True,
    "Enron": True,
    "Epinions": True,
    "Slashdot": True,
    "CondMat": False,
    "citHepTh": True,
}

LATEX_NAMES = {
    "NetHept": "NetHEPT",
    "WikiVote": "WikiVote",
    "Enron": "Enron",
    "Epinions": "Epinions",
    "Slashdot": "Slashdot",
    "CondMat": "CondMat",
    "citHepTh": "cit-HepTh",
}


def extract_graph(obj):
    """Extract an ICGraph-like object from common loader return formats."""
    required = ("n", "src", "dst")

    if all(hasattr(obj, attr) for attr in required):
        return obj

    if isinstance(obj, dict):
        for key in ("graph", "ic_graph", "G"):
            if key in obj:
                candidate = obj[key]
                if all(hasattr(candidate, attr) for attr in required):
                    return candidate

    if isinstance(obj, (tuple, list)):
        for candidate in obj:
            if all(hasattr(candidate, attr) for attr in required):
                return candidate

    raise TypeError(
        "Could not find an ICGraph-like object in load_dataset() output. "
        "Expected an object with attributes n, src, and dst."
    )


def binary_directed_adjacency(graph) -> sp.csr_matrix:
    """Simple directed adjacency matrix, with self-loops removed."""
    n = int(graph.n)
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)

    keep = src != dst
    src = src[keep]
    dst = dst[keep]

    data = np.ones(len(src), dtype=np.uint8)
    A = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()

    if A.nnz:
        A.data[:] = 1

    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    return A


def undirected_projection(A: sp.csr_matrix) -> sp.csr_matrix:
    """Simple undirected projection."""
    U = A.maximum(A.T).tocsr()
    if U.nnz:
        U.data[:] = 1
    U.setdiag(0)
    U.eliminate_zeros()
    U.sort_indices()
    return U


def nx_graph_from_sparse_undirected(U: sp.csr_matrix) -> nx.Graph:
    """Build an nx.Graph from the upper triangle only."""
    n = U.shape[0]
    upper = sp.triu(U, k=1, format="coo")

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(upper.row.tolist(), upper.col.tolist()))
    return G


def triangle_and_clustering_stats(G: nx.Graph):
    """
    Enumerate triangles once and derive both total triangles and mean
    clustering from the per-node counts.
    """
    tri_by_node = nx.triangles(G)
    n_triangles = int(sum(tri_by_node.values()) // 3)

    clustering_sum = 0.0
    n = G.number_of_nodes()

    for u, deg in G.degree():
        deg = int(deg)
        if deg >= 2:
            clustering_sum += (
                2.0 * float(tri_by_node[u]) / (deg * (deg - 1))
            )

    avg_clustering = clustering_sum / n if n else float("nan")
    return n_triangles, avg_clustering


def reciprocity_from_sparse(A: sp.csr_matrix) -> float:
    """Fraction of directed arcs whose reverse arc also exists."""
    if A.nnz == 0:
        return float("nan")
    reciprocal_arcs = A.multiply(A.T).nnz
    return float(reciprocal_arcs) / float(A.nnz)


def compute_statistics(dataset: str) -> dict:
    print(f"Loading {dataset} ...", flush=True)
    graph = extract_graph(load_dataset(dataset))

    A = binary_directed_adjacency(graph)
    U = undirected_projection(A)

    n = int(A.shape[0])
    directed = bool(IS_DIRECTED[dataset])

    directed_arcs = int(A.nnz)
    undirected_edges = int(U.nnz // 2)

    # Report source-network edges rather than internal bidirected arcs for
    # undirected datasets.
    m_reported = directed_arcs if directed else undirected_edges

    undirected_degree = np.diff(U.indptr).astype(np.float64)
    avg_degree = float(undirected_degree.mean()) if n else float("nan")

    print(
        f"  n={n:,}, reported edges={m_reported:,}, "
        f"projection edges={undirected_edges:,}",
        flush=True,
    )
    print("  counting triangles / clustering ...", flush=True)

    G = nx_graph_from_sparse_undirected(U)
    n_triangles, avg_clustering = triangle_and_clustering_stats(G)

    reciprocity = (
        reciprocity_from_sparse(A)
        if directed
        else float("nan")
    )

    del G

    return {
        "dataset": dataset,
        "nodes": n,
        "edges": m_reported,
        "directed": directed,
        "average_degree": avg_degree,
        "average_clustering": avg_clustering,
        "triangles": n_triangles,
        "reciprocity": reciprocity,
        # Diagnostics retained in the CSV.
        "stored_directed_arcs": directed_arcs,
        "undirected_projection_edges": undirected_edges,
    }


def fmt_int(x) -> str:
    return f"{int(x):,}"


def fmt_float(x, digits=3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def make_latex_table(df: pd.DataFrame, include_reciprocity: bool = True) -> str:
    by_name = df.set_index("dataset")
    header_names = [LATEX_NAMES[d] for d in DATASETS]

    rows = [
        ("Nodes", [fmt_int(by_name.loc[d, "nodes"]) for d in DATASETS]),
        ("Edges", [fmt_int(by_name.loc[d, "edges"]) for d in DATASETS]),
        (
            "Directed",
            [
                "True" if bool(by_name.loc[d, "directed"]) else "False"
                for d in DATASETS
            ],
        ),
        (
            "Average degree",
            [
                fmt_float(by_name.loc[d, "average_degree"], 2)
                for d in DATASETS
            ],
        ),
        (
            "Average clustering coefficient",
            [
                fmt_float(by_name.loc[d, "average_clustering"], 3)
                for d in DATASETS
            ],
        ),
        (
            "Triangles",
            [fmt_int(by_name.loc[d, "triangles"]) for d in DATASETS],
        ),
    ]

    if include_reciprocity:
        rows.append(
            (
                "Reciprocity",
                [
                    fmt_float(by_name.loc[d, "reciprocity"], 3)
                    if bool(by_name.loc[d, "directed"])
                    else "--"
                    for d in DATASETS
                ],
            )
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Structural properties of the real-world benchmark "
            r"networks. For directed datasets, the edge count refers to "
            r"directed arcs. Average degree, clustering, and triangle counts "
            r"are computed on the simple undirected projection. Reciprocity "
            r"is the fraction of directed arcs whose reverse arc is also "
            r"present.}"
        ),
        r"\label{tab:real-world-network-statistics}",
        r"\begin{tabular}{l" + "r" * len(DATASETS) + "}",
        r"\toprule",
        " & " + " & ".join(header_names) + r" \\",
        r"\midrule",
    ]

    for label, values in rows:
        lines.append(label + " & " + " & ".join(values) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("network_statistics_table.tex"),
        help="Output LaTeX file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("network_statistics.csv"),
        help="Output CSV containing raw statistics.",
    )
    parser.add_argument(
        "--no-reciprocity",
        action="store_true",
        help="Do not include the optional reciprocity row.",
    )
    args = parser.parse_args()

    rows = [compute_statistics(dataset) for dataset in DATASETS]
    df = pd.DataFrame(rows)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    latex = make_latex_table(
        df,
        include_reciprocity=not args.no_reciprocity,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex + "\n", encoding="utf-8")

    print("\n" + latex)
    print(f"\nSaved LaTeX table to: {args.output}")
    print(f"Saved raw statistics to: {args.csv}")


if __name__ == "__main__":
    main()
