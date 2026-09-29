from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
from collections import deque
import heapq

import numpy as np

try:
    from numba import njit, prange

    @njit(cache=False, parallel=True)
    def _build_reverse_numba(src, dst, out_indptr):
        m = src.size
        rev = np.full(m, -1, dtype=np.int64)
        for e in prange(m):
            u = int(src[e])
            v = int(dst[e])
            lo = int(out_indptr[v])
            hi = int(out_indptr[v + 1])
            # dst is sorted inside each source block.
            while lo < hi:
                mid = (lo + hi) // 2
                x = int(dst[mid])
                if x < u:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < int(out_indptr[v + 1]) and int(dst[lo]) == u:
                rev[e] = lo
        return rev

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


# -----------------------------------------------------------------------------
# Compact graph representation
# -----------------------------------------------------------------------------


def _choose_node_dtype(n: int) -> np.dtype:
    if n <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _choose_edge_dtype(m: int) -> np.dtype:
    if m <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


@dataclass(slots=True)
class ICGraph:
    """Compact directed simple graph for Independent-Cascade computations.

    Canonical representation
    ------------------------
    Edges are sorted lexicographically by (src, dst).  Node IDs are dense
    integers 0, ..., n-1.  Edge probabilities are aligned with the edge IDs.

    Stored eagerly:
        n, src, dst, prob, out_indptr

    Cached lazily:
        reverse edge IDs, incoming CSR edge IDs

    This deliberately does *not* store Python objects per node/edge.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    prob: np.ndarray
    out_indptr: np.ndarray
    labels: Optional[np.ndarray] = None

    _reverse: Optional[np.ndarray] = field(default=None, repr=False)
    _in_indptr: Optional[np.ndarray] = field(default=None, repr=False)
    _in_edges: Optional[np.ndarray] = field(default=None, repr=False)

    _r2_prev: Optional[np.ndarray] = field(default=None, repr=False)
    _r2_next: Optional[np.ndarray] = field(default=None, repr=False)
    _r2_rotate: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.n = int(self.n)
        self.src = np.ascontiguousarray(self.src)
        self.dst = np.ascontiguousarray(self.dst)
        self.prob = np.ascontiguousarray(self.prob)
        self.out_indptr = np.ascontiguousarray(self.out_indptr, dtype=np.int64)

        if self.src.ndim != 1 or self.dst.ndim != 1 or self.prob.ndim != 1:
            raise ValueError("src, dst, and prob must be one-dimensional arrays.")
        if not (len(self.src) == len(self.dst) == len(self.prob)):
            raise ValueError("src, dst, and prob must have the same length.")
        if len(self.out_indptr) != self.n + 1:
            raise ValueError("out_indptr must have length n + 1.")
        if self.n < 0:
            raise ValueError("n must be non-negative.")
        if self.m:
            if self.src.min() < 0 or self.dst.min() < 0:
                raise ValueError("Node IDs must be non-negative.")
            if self.src.max() >= self.n or self.dst.max() >= self.n:
                raise ValueError("src/dst contain a node ID outside [0, n).")
        if np.any((self.prob < 0.0) | (self.prob > 1.0)):
            raise ValueError("Edge probabilities must lie in [0, 1].")
        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError("labels must have length n.")

    @property
    def m(self) -> int:
        return int(self.src.size)

    @property
    def node_dtype(self) -> np.dtype:
        return self.src.dtype

    @property
    def edge_dtype(self) -> np.dtype:
        return _choose_edge_dtype(self.m)

    @classmethod
    def from_edges(
        cls,
        n: int,
        src: Sequence[int] | np.ndarray,
        dst: Sequence[int] | np.ndarray,
        prob: Sequence[float] | np.ndarray,
        *,
        prob_dtype: np.dtype | type = np.float32,
        labels: Optional[Sequence[Any] | np.ndarray] = None,
        check_duplicates: bool = True,
    ) -> "ICGraph":
        """Build a graph from dense integer node IDs.

        The input edge arrays may be in any order. They are sorted once by
        (src, dst), after which outgoing edge ranges are contiguous.
        """
        n = int(n)
        node_dtype = _choose_node_dtype(n)

        src_arr = np.asarray(src, dtype=node_dtype)
        dst_arr = np.asarray(dst, dtype=node_dtype)
        p_arr = np.asarray(prob, dtype=prob_dtype)

        if not (src_arr.ndim == dst_arr.ndim == p_arr.ndim == 1):
            raise ValueError("src, dst, and prob must be 1-D.")
        if not (len(src_arr) == len(dst_arr) == len(p_arr)):
            raise ValueError("src, dst, and prob must have equal length.")

        m = len(src_arr)
        if m:
            # Lexicographic sorting makes each source block contiguous *and*
            # each destination sorted inside the source block.  The latter is
            # useful for binary-search edge lookup without a Python dict.
            order = np.lexsort((dst_arr, src_arr))
            src_arr = np.ascontiguousarray(src_arr[order])
            dst_arr = np.ascontiguousarray(dst_arr[order])
            p_arr = np.ascontiguousarray(p_arr[order])

            if check_duplicates and m > 1:
                dup = (src_arr[1:] == src_arr[:-1]) & (dst_arr[1:] == dst_arr[:-1])
                if np.any(dup):
                    i = int(np.flatnonzero(dup)[0])
                    raise ValueError(
                        f"Parallel/duplicate directed edge ({int(src_arr[i])}, "
                        f"{int(dst_arr[i])}) is not supported."
                    )

        out_degree = np.bincount(src_arr.astype(np.int64, copy=False), minlength=n)
        out_indptr = np.empty(n + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(out_degree, out=out_indptr[1:])

        label_arr = None if labels is None else np.asarray(labels)
        return cls(n, src_arr, dst_arr, p_arr, out_indptr, label_arr)

    @classmethod
    def from_networkx(
        cls,
        G,
        edge_probs=None,
        *,
        prob_attr: Optional[str] = None,
        default_prob: float = 0.0,
        prob_dtype: np.dtype | type = np.float32,
        keep_labels: bool = True,
    ) -> "ICGraph":
        """Convert a NetworkX graph once, intended mainly for loading/debugging.

        For an undirected NetworkX graph, both directions are emitted because
        IC propagation is directed internally.
        """
        nodes = list(G.nodes())
        idx = {u: i for i, u in enumerate(nodes)}

        src = []
        dst = []
        p = []

        def get_prob(u, v, data):
            if edge_probs is not None:
                if (u, v) in edge_probs:
                    return float(edge_probs[(u, v)])
                if not G.is_directed() and (v, u) in edge_probs:
                    return float(edge_probs[(v, u)])
                return float(default_prob)
            if prob_attr is not None:
                return float(data.get(prob_attr, default_prob))
            return float(default_prob)

        for u, v, data in G.edges(data=True):
            src.append(idx[u])
            dst.append(idx[v])
            p.append(get_prob(u, v, data))
            if not G.is_directed() and u != v:
                src.append(idx[v])
                dst.append(idx[u])
                p.append(get_prob(v, u, data))

        labels = np.asarray(nodes, dtype=object) if keep_labels else None
        return cls.from_edges(
            len(nodes), src, dst, p,
            prob_dtype=prob_dtype,
            labels=labels,
        )

    def out_edge_ids(self, u: int) -> np.ndarray:
        a = int(self.out_indptr[u])
        b = int(self.out_indptr[u + 1])
        # Since canonical edge IDs are source-sorted, no separate out_edges
        # array is needed.
        return np.arange(a, b, dtype=self.edge_dtype)

    def successors(self, u: int) -> np.ndarray:
        a = int(self.out_indptr[u])
        b = int(self.out_indptr[u + 1])
        return self.dst[a:b]

    def edge_id(self, u: int, v: int) -> int:
        """Return edge ID for u->v, or -1 if absent."""
        a = int(self.out_indptr[u])
        b = int(self.out_indptr[u + 1])
        if a == b:
            return -1
        block = self.dst[a:b]
        j = int(np.searchsorted(block, v))
        if j < len(block) and int(block[j]) == int(v):
            return a + j
        return -1

    def ensure_reverse(self, *, chunk_size: int = 5_000_000) -> np.ndarray:
        """Build reverse[e] = edge ID of dst[e]->src[e], or -1.

        Uses binary searches in source-sorted adjacency blocks and processes
        edges in chunks, avoiding an O(m) Python dictionary.
        """
        if self._reverse is not None:
            return self._reverse

        edge_dtype = self.edge_dtype
        rev = np.full(self.m, -1, dtype=edge_dtype)
        if self.m == 0:
            self._reverse = rev
            return rev

        if _NUMBA_AVAILABLE:
            # No Python dictionary and no O(m) int64 key array.  This is the
            # preferred path for large graphs.
            rev64 = _build_reverse_numba(self.src, self.dst, self.out_indptr)
            rev = rev64.astype(edge_dtype, copy=False)
            self._reverse = rev
            return rev

        # Dependency-free fallback. It uses a temporary int64 key array and is
        # therefore less attractive at very large m, but keeps the module usable
        # without Numba.
        if self.n > 3_037_000_499:  # floor(sqrt(int64 max))
            for e in range(self.m):
                rev[e] = self.edge_id(int(self.dst[e]), int(self.src[e]))
            self._reverse = rev
            return rev

        keys = self.src.astype(np.int64) * self.n + self.dst.astype(np.int64)
        for start in range(0, self.m, chunk_size):
            stop = min(start + chunk_size, self.m)
            rkeys = self.dst[start:stop].astype(np.int64) * self.n + self.src[start:stop].astype(np.int64)
            pos = np.searchsorted(keys, rkeys)
            valid = pos < self.m
            pos_safe = np.minimum(pos, self.m - 1)
            valid &= keys[pos_safe] == rkeys
            local = rev[start:stop]
            local[valid] = pos_safe[valid].astype(edge_dtype, copy=False)

        self._reverse = rev
        return rev

    def ensure_incoming(self) -> tuple[np.ndarray, np.ndarray]:
        """Build CSR-like incoming edge IDs lazily.

        Returns
        -------
        in_indptr : int64[n+1]
        in_edges  : edge IDs sorted/grouped by destination
        """
        if self._in_indptr is not None and self._in_edges is not None:
            return self._in_indptr, self._in_edges

        edge_dtype = self.edge_dtype
        if self.m == 0:
            self._in_indptr = np.zeros(self.n + 1, dtype=np.int64)
            self._in_edges = np.empty(0, dtype=edge_dtype)
            return self._in_indptr, self._in_edges

        order = np.argsort(self.dst, kind="stable")
        in_edges = order.astype(edge_dtype, copy=False)
        indegree = np.bincount(self.dst.astype(np.int64, copy=False), minlength=self.n)
        in_indptr = np.empty(self.n + 1, dtype=np.int64)
        in_indptr[0] = 0
        np.cumsum(indegree, out=in_indptr[1:])

        self._in_indptr = in_indptr
        self._in_edges = np.ascontiguousarray(in_edges)
        return self._in_indptr, self._in_edges

    def in_edge_ids(self, v: int) -> np.ndarray:
        indptr, edges = self.ensure_incoming()
        a = int(indptr[v])
        b = int(indptr[v + 1])
        return edges[a:b]

    def predecessors(self, v: int) -> np.ndarray:
        e = self.in_edge_ids(v)
        return self.src[e]

    def memory_bytes(self, *, include_cached: bool = True) -> int:
        arrays = [self.src, self.dst, self.prob, self.out_indptr]
        if self.labels is not None:
            # For object labels, nbytes only counts pointers, not Python objects.
            arrays.append(self.labels)
        if include_cached:
            for x in (
                self._reverse,
                self._in_indptr,
                self._in_edges,
                self._r2_prev,
                self._r2_next,
                self._r2_rotate,
            ):
                if x is not None:
                    arrays.append(x)
        return int(sum(x.nbytes for x in arrays))

    def memory_gb(self, *, include_cached: bool = True) -> float:
        return self.memory_bytes(include_cached=include_cached) / 1e9

    def as_node_dict(self, values: Sequence[float] | np.ndarray) -> dict:
        """Convert an aligned node array back to original labels when available."""
        values = np.asarray(values)
        if len(values) != self.n:
            raise ValueError("values must have length n.")
        if self.labels is None:
            return {i: values[i] for i in range(self.n)}
        return {self.labels[i]: values[i] for i in range(self.n)}

    def _build_r2_triangles_hash(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enumerate the triangle-conditioned states used by dmp_est_r2
        using a temporary hash table for O(1)-average edge lookup.
    
        The temporary hash table is discarded once this method returns.
    
        Returns
        -------
        triangle_prev : ndarray
            For state m2[w | v,u], edge ID of w -> v.
    
        triangle_next : ndarray
            Edge ID of v -> u whose m1 update receives the r=2 correction.
    
        triangle_rotate : ndarray
            Maps each r=2 state to the rotated state whose factor must be
            removed when computing the second-order cavity product.
    
        Notes
        -----
        A directed triangle
    
            w -> v -> u -> w
    
        generates three states:
    
            m2[w | v,u]
            m2[u | w,v]
            m2[v | u,w]
    
        Each directed triangle is processed exactly once, using its
        smallest edge ID as the canonical representative.
        """
        from array import array
    
        edge_dtype = self.edge_dtype
    
        if self.m == 0:
            empty = np.empty(0, dtype=edge_dtype)
            return empty, empty.copy(), empty.copy()
    
        if np.any(self.src == self.dst):
            raise ValueError(
                "r=2 triangle construction does not support self-loops."
            )
    
        # r=2 enumeration needs incoming edge lists.
        in_indptr, in_edges = self.ensure_incoming()
    
        # Compact temporary buffers. Avoid Python lists containing millions
        # of boxed integers.
        if edge_dtype == np.dtype(np.int32):
            typecode = "i"
            np_dtype = np.int32
        else:
            typecode = "q"
            np_dtype = np.int64
    
        triangle_prev_buffer = array(typecode)
        triangle_next_buffer = array(typecode)
        triangle_rotate_buffer = array(typecode)
    
        src = self.src
        dst = self.dst
        out_indptr = self.out_indptr
        n = self.n
    
        # ----------------------------------------------------------
        # Temporary edge lookup:
        #
        #     (u,v) -> edge ID
        #
        # encoded using the unique integer
        #
        #     key = u*n + v
        #
        # rather than Python tuple keys.
        # ----------------------------------------------------------
    
        pair_to_edge = {
            int(src[e]) * n + int(dst[e]): e
            for e in range(self.m)
        }
    
        # ----------------------------------------------------------
        # Enumerate directed triangles
        # ----------------------------------------------------------
    
        for edge_vu in range(self.m):
    
            # Current edge:
            #
            #     v -> u
            v = int(src[edge_vu])
            u = int(dst[edge_vu])
    
            # Incoming edges w -> v
            ia = int(in_indptr[v])
            ib = int(in_indptr[v + 1])
    
            # Outgoing edges u -> w
            oa = int(out_indptr[u])
            ob = int(out_indptr[u + 1])
    
            # Search the smaller candidate neighborhood.
            if (ib - ia) <= (ob - oa):
    
                # Enumerate w -> v and test whether u -> w exists.
                for pos in range(ia, ib):
    
                    edge_wv = int(in_edges[pos])
                    w = int(src[edge_wv])
    
                    edge_uw = pair_to_edge.get(
                        u * n + w,
                        -1,
                    )
    
                    if edge_uw < 0:
                        continue
    
                    # The triangle is:
                    #
                    #     w -> v -> u -> w
                    #
                    # It will be discovered once from each of its three
                    # edges. Retain only the occurrence where edge_vu
                    # has the smallest edge ID.
                    if not (
                        edge_vu < edge_wv
                        and edge_vu < edge_uw
                    ):
                        continue
    
                    base = len(triangle_prev_buffer)
    
                    # A: m2[w | v,u]
                    triangle_prev_buffer.append(edge_wv)
                    triangle_next_buffer.append(edge_vu)
    
                    # B: m2[u | w,v]
                    triangle_prev_buffer.append(edge_uw)
                    triangle_next_buffer.append(edge_wv)
    
                    # C: m2[v | u,w]
                    triangle_prev_buffer.append(edge_vu)
                    triangle_next_buffer.append(edge_uw)
    
                    # A removes B's factor,
                    # B removes C's factor,
                    # C removes A's factor.
                    triangle_rotate_buffer.extend(
                        (
                            base + 1,
                            base + 2,
                            base,
                        )
                    )
    
            else:
    
                # Enumerate u -> w and test whether w -> v exists.
                for edge_uw in range(oa, ob):
    
                    w = int(dst[edge_uw])
    
                    edge_wv = pair_to_edge.get(
                        w * n + v,
                        -1,
                    )
    
                    if edge_wv < 0:
                        continue
    
                    if not (
                        edge_vu < edge_wv
                        and edge_vu < edge_uw
                    ):
                        continue
    
                    base = len(triangle_prev_buffer)
    
                    triangle_prev_buffer.append(edge_wv)
                    triangle_next_buffer.append(edge_vu)
    
                    triangle_prev_buffer.append(edge_uw)
                    triangle_next_buffer.append(edge_wv)
    
                    triangle_prev_buffer.append(edge_vu)
                    triangle_next_buffer.append(edge_uw)
    
                    triangle_rotate_buffer.extend(
                        (
                            base + 1,
                            base + 2,
                            base,
                        )
                    )
    
        # pair_to_edge disappears here after return. Its gluttonous Python
        # existence was temporary and therefore morally acceptable.
    
        triangle_prev = np.frombuffer(
            triangle_prev_buffer,
            dtype=np_dtype,
        ).copy()
    
        triangle_next = np.frombuffer(
            triangle_next_buffer,
            dtype=np_dtype,
        ).copy()
    
        triangle_rotate = np.frombuffer(
            triangle_rotate_buffer,
            dtype=np_dtype,
        ).copy()
    
        return (
            triangle_prev,
            triangle_next,
            triangle_rotate,
        )

    def _build_r2_triangles_binary(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enumerate the triangle-conditioned states used by dmp_est_r2
        without constructing an O(m) Python hash table.
    
        Edge existence is tested using ICGraph.edge_id(), which performs
        binary search inside the source node's sorted adjacency block.
    
        This is more memory-efficient than _build_r2_triangles_hash(),
        but generally slower on small/medium graphs.
        """
        from array import array
    
        edge_dtype = self.edge_dtype
    
        if self.m == 0:
            empty = np.empty(0, dtype=edge_dtype)
            return empty, empty.copy(), empty.copy()
    
        if np.any(self.src == self.dst):
            raise ValueError(
                "r=2 triangle construction does not support self-loops."
            )
    
        in_indptr, in_edges = self.ensure_incoming()
    
        if edge_dtype == np.dtype(np.int32):
            typecode = "i"
            np_dtype = np.int32
        else:
            typecode = "q"
            np_dtype = np.int64
    
        triangle_prev_buffer = array(typecode)
        triangle_next_buffer = array(typecode)
        triangle_rotate_buffer = array(typecode)
    
        src = self.src
        dst = self.dst
        out_indptr = self.out_indptr
    
        # ----------------------------------------------------------
        # Enumerate directed triangles
        # ----------------------------------------------------------
    
        for edge_vu in range(self.m):
    
            v = int(src[edge_vu])
            u = int(dst[edge_vu])
    
            # Incoming w -> v
            ia = int(in_indptr[v])
            ib = int(in_indptr[v + 1])
    
            # Outgoing u -> w
            oa = int(out_indptr[u])
            ob = int(out_indptr[u + 1])
    
            # Again, iterate through whichever candidate set is smaller.
            if (ib - ia) <= (ob - oa):
    
                for pos in range(ia, ib):
    
                    edge_wv = int(in_edges[pos])
                    w = int(src[edge_wv])
    
                    # Binary search for u -> w.
                    edge_uw = self.edge_id(u, w)
    
                    if edge_uw < 0:
                        continue
    
                    if not (
                        edge_vu < edge_wv
                        and edge_vu < edge_uw
                    ):
                        continue
    
                    base = len(triangle_prev_buffer)
    
                    triangle_prev_buffer.append(edge_wv)
                    triangle_next_buffer.append(edge_vu)
    
                    triangle_prev_buffer.append(edge_uw)
                    triangle_next_buffer.append(edge_wv)
    
                    triangle_prev_buffer.append(edge_vu)
                    triangle_next_buffer.append(edge_uw)
    
                    triangle_rotate_buffer.extend(
                        (
                            base + 1,
                            base + 2,
                            base,
                        )
                    )
    
            else:
    
                for edge_uw in range(oa, ob):
    
                    w = int(dst[edge_uw])
    
                    # Binary search for w -> v.
                    edge_wv = self.edge_id(w, v)
    
                    if edge_wv < 0:
                        continue
    
                    if not (
                        edge_vu < edge_wv
                        and edge_vu < edge_uw
                    ):
                        continue
    
                    base = len(triangle_prev_buffer)
    
                    triangle_prev_buffer.append(edge_wv)
                    triangle_next_buffer.append(edge_vu)
    
                    triangle_prev_buffer.append(edge_uw)
                    triangle_next_buffer.append(edge_wv)
    
                    triangle_prev_buffer.append(edge_vu)
                    triangle_next_buffer.append(edge_uw)
    
                    triangle_rotate_buffer.extend(
                        (
                            base + 1,
                            base + 2,
                            base,
                        )
                    )
    
        triangle_prev = np.frombuffer(
            triangle_prev_buffer,
            dtype=np_dtype,
        ).copy()
    
        triangle_next = np.frombuffer(
            triangle_next_buffer,
            dtype=np_dtype,
        ).copy()
    
        triangle_rotate = np.frombuffer(
            triangle_rotate_buffer,
            dtype=np_dtype,
        ).copy()
    
        return (
            triangle_prev,
            triangle_next,
            triangle_rotate,
        )
    
    def ensure_r2_triangles(
        self,
        *,
        lookup: str = "auto",
        hash_max_edges: int = 5_000_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return cached r=2 triangle-state indexing.
    
        Parameters
        ----------
        lookup : {"auto", "hash", "binary"}
            "hash"
                Temporary Python hash table. Faster, more temporary RAM.
    
            "binary"
                Binary-search edge lookup. Lower temporary RAM.
    
            "auto"
                Use hash lookup when m <= hash_max_edges and binary
                search otherwise.
    
        hash_max_edges : int
            Directed-edge threshold used by lookup="auto".
        """
    
        if (
            self._r2_prev is not None
            and self._r2_next is not None
            and self._r2_rotate is not None
        ):
            return (
                self._r2_prev,
                self._r2_next,
                self._r2_rotate,
            )
    
        if lookup == "auto":
            lookup = (
                "hash"
                if self.m <= hash_max_edges
                else "binary"
            )
    
        if lookup == "hash":
            triangles = self._build_r2_triangles_hash()
    
        elif lookup == "binary":
            triangles = self._build_r2_triangles_binary()
    
        else:
            raise ValueError(
                'lookup must be "auto", "hash", or "binary".'
            )
    
        (
            self._r2_prev,
            self._r2_next,
            self._r2_rotate,
        ) = triangles
    
        return triangles

def _prior_array(graph: ICGraph, prior_probs, dtype=None) -> np.ndarray:
    """Normalize priors to a dense array aligned with graph node IDs."""
    if dtype is None:
        dtype = np.result_type(graph.prob.dtype, np.float32)

    if isinstance(prior_probs, dict):
        if graph.labels is None:
            p0 = np.array([prior_probs.get(i, 0.0) for i in range(graph.n)], dtype=dtype)
        else:
            p0 = np.array([prior_probs.get(graph.labels[i], 0.0) for i in range(graph.n)], dtype=dtype)
    else:
        p0 = np.asarray(prior_probs, dtype=dtype).reshape(-1)
        if len(p0) != graph.n:
            raise ValueError("prior_probs must have length graph.n.")
        p0 = p0.copy()

    return np.clip(p0, 0.0, 1.0)


def _incoming_log_product(graph: ICGraph, q_e: np.ndarray) -> np.ndarray:
    """For each node v, compute prod_{e: dst[e]=v} q_e[e]."""
    log_sum = np.bincount(
        graph.dst.astype(np.int64, copy=False),
        weights=np.log(q_e),
        minlength=graph.n,
    )
    return np.exp(log_sum)


def _directed_multi_source_bfs_dist(graph: ICGraph, sources: np.ndarray) -> np.ndarray:
    from collections import deque

    dist = np.full(graph.n, -1, dtype=np.int64)
    q = deque()

    for s in np.asarray(sources, dtype=np.int64):
        s = int(s)
        if 0 <= s < graph.n and dist[s] < 0:
            dist[s] = 0
            q.append(s)

    while q:
        u = q.popleft()
        nd = dist[u] + 1
        a = int(graph.out_indptr[u])
        b = int(graph.out_indptr[u + 1])
        for v in graph.dst[a:b]:
            v = int(v)
            if dist[v] < 0:
                dist[v] = nd
                q.append(v)

    return dist


# -----------------------------------------------------------------------------
# IC Monte Carlo
# -----------------------------------------------------------------------------


def optimized_independent_cascade(
    graph: ICGraph,
    prior_probs,
    k: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Monte-Carlo IC simulation; returns node marginals aligned with node IDs."""
    rng = np.random.default_rng(seed)
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)

    infection_counts = np.zeros(graph.n, dtype=np.int64)
    out_degree = np.diff(graph.out_indptr)

    for _ in range(int(k)):
        active = rng.random(graph.n) < p0
        visited = active.copy()
        infection_counts += active
        frontier = np.flatnonzero(active)

        while frontier.size:
            degs = out_degree[frontier]
            total_edges = int(degs.sum())
            if total_edges == 0:
                break

            repeated_frontier = np.repeat(frontier, degs)
            starts = graph.out_indptr[repeated_frontier]
            group_starts = np.repeat(np.cumsum(degs) - degs, degs)
            offsets = np.arange(total_edges, dtype=np.int64) - group_starts
            edge_idx = starts + offsets

            candidate_dst = graph.dst[edge_idx]
            valid = ~visited[candidate_dst]
            if not np.any(valid):
                break

            edge_idx = edge_idx[valid]
            candidate_dst = graph.dst[edge_idx]
            success = rng.random(candidate_dst.size) < graph.prob[edge_idx]
            if not np.any(success):
                break

            activated_nodes = np.unique(candidate_dst[success])
            frontier = activated_nodes[~visited[activated_nodes]]
            if frontier.size == 0:
                break

            visited[frontier] = True
            infection_counts[frontier] += 1

    return infection_counts / float(k)


# -----------------------------------------------------------------------------
# DMP variants
# -----------------------------------------------------------------------------


def dmp_est(graph: ICGraph, prior_probs, T: int, eps: float = 1e-20) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    p0 = _prior_array(graph, prior_probs, dtype=dtype)
    if graph.m == 0:
        return p0

    src, dst = graph.src, graph.dst
    p_e = graph.prob.astype(dtype, copy=False)
    rev = graph.ensure_reverse()

    m_e = p0[src].copy()
    for _ in range(int(T)):
        q_e = np.clip(1.0 - p_e * m_e, eps, 1.0)
        prod_incoming = _incoming_log_product(graph, q_e).astype(dtype, copy=False)

        prod_excl = prod_incoming[src].copy()
        mask = rev >= 0
        prod_excl[mask] /= q_e[rev[mask]]
        prod_excl = np.clip(prod_excl, 0.0, 1.0)

        m_e = 1.0 - (1.0 - p0[src]) * prod_excl
        np.clip(m_e, 0.0, 1.0, out=m_e)

    q_e = np.clip(1.0 - p_e * m_e, eps, 1.0)
    prod_incoming = _incoming_log_product(graph, q_e)
    pi = 1.0 - (1.0 - p0) * prod_incoming
    return np.clip(pi, 0.0, 1.0)


def dmp_inf(
    graph: ICGraph,
    prior_probs,
    eps: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    p0 = _prior_array(graph, prior_probs, dtype=dtype)
    if graph.m == 0:
        return p0

    src = graph.src
    p_e = graph.prob.astype(dtype, copy=False)
    rev = graph.ensure_reverse()
    m_e = p0[src].copy()

    for _ in range(int(max_iter)):
        q_e = np.clip(1.0 - p_e * m_e, eps, 1.0)
        prod_incoming = _incoming_log_product(graph, q_e).astype(dtype, copy=False)

        prod_excl = prod_incoming[src].copy()
        mask = rev >= 0
        prod_excl[mask] /= q_e[rev[mask]]
        prod_excl = np.clip(prod_excl, 0.0, 1.0)

        new_m = 1.0 - (1.0 - p0[src]) * prod_excl
        np.clip(new_m, 0.0, 1.0, out=new_m)
        delta = float(np.max(np.abs(new_m - m_e)))
        m_e = new_m
        if delta <= eps:
            break

    q_e = np.clip(1.0 - p_e * m_e, eps, 1.0)
    prod_incoming = _incoming_log_product(graph, q_e)
    pi = 1.0 - (1.0 - p0) * prod_incoming
    return np.clip(pi, 0.0, 1.0)


def dmp_python(graph: ICGraph, prior_probs, T: int = 10) -> np.ndarray:
    """Readable reference implementation of standard DMP on ICGraph.

    Unlike the old NetworkX reference code, this explicitly uses incoming
    neighbors, matching the recurrence implemented by dmp_est.
    """
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)
    rev = graph.ensure_reverse()
    in_indptr, in_edges = graph.ensure_incoming()
    m = p0[graph.src].copy()

    for _ in range(int(T)):
        new_m = np.empty_like(m)
        for e in range(graph.m):
            u = int(graph.src[e])
            a, b = int(in_indptr[u]), int(in_indptr[u + 1])
            influence = 1.0
            for f in in_edges[a:b]:
                f = int(f)
                if rev[e] == f:
                    continue
                influence *= 1.0 - float(graph.prob[f]) * m[f]
            new_m[e] = 1.0 - (1.0 - p0[u]) * influence
        m = new_m

    p = np.empty(graph.n, dtype=np.float64)
    for u in range(graph.n):
        a, b = int(in_indptr[u]), int(in_indptr[u + 1])
        influence = 1.0
        for f in in_edges[a:b]:
            f = int(f)
            influence *= 1.0 - float(graph.prob[f]) * m[f]
        p[u] = 1.0 - (1.0 - p0[u]) * influence
    return np.clip(p, 0.0, 1.0)


def swe_hib_cavity(
    graph: ICGraph,
    prior_probs,
    T: int = 10,
    eps: float = 1e-12,
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    p0 = _prior_array(graph, prior_probs, dtype=dtype)
    if graph.m == 0:
        return p0

    src = graph.src
    p_e = graph.prob.astype(dtype, copy=False)
    rev = graph.ensure_reverse()

    p_wo = p0[src].copy()
    p_wo_prev = np.zeros_like(p_wo)

    for _ in range(int(T)):
        delta = p_wo - p_wo_prev
        q = np.clip(1.0 - p_e * delta, eps, 1.0)
        prod_full = _incoming_log_product(graph, q).astype(dtype, copy=False)

        prod_cavity = prod_full[src].copy()
        mask = rev >= 0
        prod_cavity[mask] /= q[rev[mask]]
        prod_cavity = np.clip(prod_cavity, eps, 1.0)

        p_wo_new = p_wo + (1.0 - p_wo) * (1.0 - prod_cavity)
        p_wo_prev, p_wo = p_wo, p_wo_new

    q_final = np.clip(1.0 - p_e * p_wo, eps, 1.0)
    prod_final = _incoming_log_product(graph, q_final)
    p = 1.0 - (1.0 - p0) * prod_final
    return np.clip(p, 0.0, 1.0)


def dmp_um_python(graph: ICGraph, prior_probs, T: int = 10) -> np.ndarray:
    """Readable reference implementation of the UM cavity update."""
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)
    rev = graph.ensure_reverse()
    in_indptr, in_edges = graph.ensure_incoming()

    p_wo = p0[graph.src].copy()
    p_wo_prev = np.zeros_like(p_wo)

    for _ in range(int(T)):
        new = np.empty_like(p_wo)
        delta = p_wo - p_wo_prev
        for e in range(graph.m):
            u = int(graph.src[e])
            a, b = int(in_indptr[u]), int(in_indptr[u + 1])
            influence = 1.0
            for f in in_edges[a:b]:
                f = int(f)
                if rev[e] == f:
                    continue
                influence *= 1.0 - float(graph.prob[f]) * delta[f]
            new[e] = p_wo[e] + (1.0 - p_wo[e]) * (1.0 - influence)
        p_wo_prev, p_wo = p_wo, new

    p = np.empty(graph.n, dtype=np.float64)
    for u in range(graph.n):
        a, b = int(in_indptr[u]), int(in_indptr[u + 1])
        influence = 1.0
        for f in in_edges[a:b]:
            f = int(f)
            influence *= 1.0 - float(graph.prob[f]) * p_wo[f]
        p[u] = 1.0 - (1.0 - p0[u]) * influence
    return np.clip(p, 0.0, 1.0)


# -----------------------------------------------------------------------------
# ALE and simple mean-field approximations
# -----------------------------------------------------------------------------


def _second_order_incoming(
    graph: ICGraph,
    state: np.ndarray,
    *,
    dtype=None,
) -> np.ndarray:
    """Second-order inclusion-exclusion approximation of noisy-OR.

    For
        z_e = p_e * state[src(e)],
    computes, for each node v,

        sum_{e -> v} z_e - sum_{e<f, e,f -> v} z_e z_f.

    The pair term is evaluated in O(|E|) time via

        sum_{i<j} z_i z_j
            = 0.5 * [ (sum_i z_i)^2 - sum_i z_i^2 ],

    so no explicit enumeration of incoming edge pairs is required.

    The returned value is the raw second-order truncation and is therefore
    not clipped to [0, 1]. Public ALE2 methods apply the same probability-style
    clipping convention used by the existing ALE implementations.
    """
    if dtype is None:
        dtype = np.result_type(graph.prob.dtype, np.float32)

    state = np.asarray(state, dtype=dtype)
    prob = graph.prob.astype(dtype, copy=False)
    dst = graph.dst.astype(np.int64, copy=False)

    edge_term = prob * state[graph.src]

    first = np.bincount(
        dst,
        weights=edge_term,
        minlength=graph.n,
    ).astype(dtype, copy=False)

    squared = np.bincount(
        dst,
        weights=edge_term * edge_term,
        minlength=graph.n,
    ).astype(dtype, copy=False)

    pair_overlap = 0.5 * (first * first - squared)

    # Roundoff can make pair_overlap microscopically negative when a node has
    # only one effective incoming contribution. The exact quantity is >= 0.
    np.maximum(pair_overlap, 0.0, out=pair_overlap)

    return first - pair_overlap


def ALE2(graph: ICGraph, prior_probs, num_steps: int) -> np.ndarray:
    """Second-order ALE.

    This mirrors ``ALE_heuristic`` but replaces the first-order additive
    incoming operator

        sum_u p_uv x_u

    by the second-order inclusion-exclusion truncation

        sum_u p_uv x_u
        - sum_{u<w} p_uv p_wv x_u x_w.

    Hence ALE2 corrects the leading pairwise overlap among simultaneous
    incoming activation possibilities while retaining ALE's walk-style
    propagation and its O(T|E|) complexity.

    ``num_steps`` follows the existing ``ALE_heuristic`` convention:
    num_steps=1 returns only the initial activation vector, num_steps=2 adds
    one propagated layer, etc.
    """
    dtype = np.result_type(graph.prob.dtype, np.float32)
    prior = _prior_array(graph, prior_probs, dtype=dtype)

    current = prior.copy()
    result = prior.copy()

    for _ in range(1, int(num_steps)):
        msg = _second_order_incoming(graph, current, dtype=dtype)
        np.clip(msg, 0.0, 1.0, out=msg)
        current = msg
        result += current

    return np.clip(result, 0.0, 1.0)


# Naming alias for code that follows the existing ALE_heuristic convention.
ALE2_heuristic = ALE2


def ALE_heuristic(graph: ICGraph, prior_probs, num_steps: int) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    x = _prior_array(graph, prior_probs, dtype=dtype)
    current_x = x.copy()
    result = x.copy()
    msg = np.zeros(graph.n, dtype=dtype)

    for _ in range(1, int(num_steps)):
        msg.fill(0.0)
        np.add.at(msg, graph.dst, current_x[graph.src] * graph.prob)
        np.clip(msg, 0.0, 1.0, out=msg)
        current_x = msg.copy()
        result += current_x

    return np.clip(result, 0.0, 1.0)

def cavity_ALE(graph, prior_probs, num_steps):
    dtype = np.result_type(graph.prob.dtype, np.float32)
    prior = _prior_array(graph, prior_probs, dtype=dtype)

    src = graph.src
    dst = graph.dst
    prob = graph.prob.astype(dtype, copy=False)

    rev = _reverse_edge_index(src, dst, graph.n)

    # h[e] = cavity mass at src[e] to be transmitted along e
    h = prior[src].copy()

    result = prior.copy()

    for _ in range(1, int(num_steps)):
        # Actual mass arriving at every node from cavity messages
        edge_contrib = prob * h
        node_msg = np.bincount(
            dst,
            weights=edge_contrib,
            minlength=graph.n
        ).astype(dtype, copy=False)

        np.clip(node_msg, 0.0, 1.0, out=node_msg)
        result += node_msg

        # For edge u -> v, start from all influence reaching u
        h_new = node_msg[src].copy()

        # Remove v -> u contribution
        mask = rev >= 0
        h_new[mask] -= edge_contrib[rev[mask]]

        np.clip(h_new, 0.0, 1.0, out=h_new)
        h = h_new

    return np.clip(result, 0.0, 1.0)

def modified_ALE(graph: ICGraph, prior_probs, num_steps: int) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    x = _prior_array(graph, prior_probs, dtype=dtype)
    survival = 1.0 - x
    msg = np.zeros(graph.n, dtype=dtype)

    for _ in range(int(num_steps)):
        msg.fill(0.0)
        np.add.at(msg, graph.dst, x[graph.src] * graph.prob)
        np.clip(msg, 0.0, 1.0, out=msg)
        survival *= 1.0 - msg
        x = msg.copy()

    return np.clip(1.0 - survival, 0.0, 1.0)


def modified_ALE2(
    graph: ICGraph,
    prior_probs,
    num_steps: int,
) -> np.ndarray:
    """Modified ALE with second-order incoming aggregation.

    This is the direct second-order analogue of ``modified_ALE``. It leaves
    the modified-ALE survival bookkeeping unchanged and replaces only the
    per-step first-order additive incoming message by the second-order
    inclusion-exclusion approximation used by :func:`ALE2`.

    In particular, if h_t denotes the raw propagation state,

        h_{t+1} = G_2(h_t),

    where

        G_2(x)_v
          = sum_{u->v} p_uv x_u
            - sum_{u<w} p_uv p_wv x_u x_w,

    while cumulative activation is accumulated through

        survival *= (1 - h_{t+1}).

    ``num_steps`` intentionally follows the existing ``modified_ALE``
    convention: each step performs one propagation update.
    """
    dtype = np.result_type(graph.prob.dtype, np.float32)
    x = _prior_array(graph, prior_probs, dtype=dtype)
    survival = 1.0 - x

    for _ in range(int(num_steps)):
        msg = _second_order_incoming(graph, x, dtype=dtype)
        np.clip(msg, 0.0, 1.0, out=msg)

        survival *= 1.0 - msg
        x = msg

    return np.clip(1.0 - survival, 0.0, 1.0)


def modified_ALE_cavity(
    graph: ICGraph,
    prior_probs,
    num_steps: int
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)

    prior = _prior_array(graph, prior_probs, dtype=dtype)

    src = graph.src
    dst = graph.dst
    prob = graph.prob.astype(dtype, copy=False)

    # rev[e] = index of reverse edge, or -1 if none exists.
    rev = _reverse_edge_index(src, dst, graph.n)

    # h[e] is the cavity mass at src[e] that may propagate along edge e.
    h = prior[src].copy()

    survival = 1.0 - prior

    node_msg_raw = np.zeros(graph.n, dtype=dtype)

    for _ in range(int(num_steps)):
        # Contribution sent along every directed edge.
        edge_msg = h * prob

        # Total new ALE mass arriving at each node.
        node_msg_raw.fill(0.0)
        np.add.at(node_msg_raw, dst, edge_msg)

        # Same probability-style clipping as modified_ALE.
        node_msg = np.clip(node_msg_raw, 0.0, 1.0)

        # Accumulate probability of never having activated.
        survival *= 1.0 - node_msg

        # ---------------------------------------------------------
        # Construct cavity states for the next propagation step.
        #
        # For edge u -> v:
        #
        #   h_new[u -> v]
        #       = sum_{w -> u, w != v} p_wu h[w -> u]
        #
        # Start with all mass arriving at u...
        # ---------------------------------------------------------
        h_new = node_msg_raw[src].copy()

        # ...then remove the contribution from v -> u.
        mask = rev >= 0
        h_new[mask] -= edge_msg[rev[mask]]

        # Preserve the modified-ALE probability-like interpretation.
        np.clip(h_new, 0.0, 1.0, out=h_new)

        h = h_new

    return np.clip(1.0 - survival, 0.0, 1.0)

def Naive(graph: ICGraph, prior_probs, T: int, eps: float = 1e-12) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    prior = _prior_array(graph, prior_probs, dtype=dtype)
    p = prior.copy()

    for _ in range(int(T)):
        contrib = np.clip(graph.prob * p[graph.src], 0.0, 1.0 - eps)
        q = np.clip(1.0 - contrib, eps, 1.0)
        incoming_prod = _incoming_log_product(graph, q)
        p = 1.0 - (1.0 - prior) * incoming_prod

    return np.clip(p, 0.0, 1.0)


# -----------------------------------------------------------------------------
# swe_no / UM
# -----------------------------------------------------------------------------


def swe_no(
    graph: ICGraph,
    prior_probs,
    T: int,
    a: float = 1.0,
    layers: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    P = _prior_array(graph, prior_probs, dtype=dtype)
    p0 = P.copy()
    p_e = np.clip(graph.prob.astype(dtype, copy=False) * a, 0.0, 1.0)
    product_term = np.ones(graph.n, dtype=dtype)

    for _ in range(1, int(T) + 1):
        vals = np.clip(p_e * P[graph.src], 0.0, 1.0 - eps)
        log_prod = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=np.log1p(-vals),
            minlength=graph.n,
        )
        prod_term = np.exp(log_prod)
        product_term *= 1.0 - P
        P = product_term * (1.0 - prod_term)

    probs = 1.0 - product_term * (1.0 - P)

    for _ in range(int(layers)):
        vals = np.clip(p_e * probs[graph.src], 0.0, 1.0 - eps)
        log_prod = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=np.log1p(-vals),
            minlength=graph.n,
        )
        neighbor_term = np.exp(log_prod)
        probs = 1.0 - (1.0 - p0) * neighbor_term

    return np.clip(probs, 0.0, 1.0)

def additive_swe(graph: ICGraph, prior_probs, num_steps: int) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)

    r = _prior_array(graph, prior_probs, dtype=dtype)
    cumulative = r.copy()

    msg = np.zeros(graph.n, dtype=dtype)

    for _ in range(int(num_steps)):
        msg.fill(0.0)
        np.add.at(msg, graph.dst, r[graph.src] * graph.prob)

        r = (1.0 - cumulative) * msg
        cumulative += r

    return np.clip(cumulative, 0.0, 1.0)

def additive_swe_cavity(
    graph: ICGraph,
    prior_probs,
    num_steps: int
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)

    prior = _prior_array(graph, prior_probs, dtype=dtype)

    src = graph.src
    dst = graph.dst
    prob = graph.prob.astype(dtype, copy=False)

    # rev[e] = index of reverse edge, or -1 if it does not exist.
    rev = _reverse_edge_index(src, dst, graph.n)
    has_rev = rev >= 0

    # rho[e]:
    # probability mass newly active at src[e] in the cavity excluding dst[e].
    rho = prior[src].copy()

    # Cumulative cavity activation probability.
    cavity_cumulative = rho.copy()

    # Actual node-level cumulative activation probability.
    cumulative = prior.copy()

    for _ in range(int(num_steps)):
        # ----------------------------------------------------------
        # Contributions generated by the current cavity states.
        # For edge f = w -> u:
        #
        #   edge_msg[f] = p_wu * rho_{w->u}(t-1)
        # ----------------------------------------------------------
        edge_msg = prob * rho

        # ----------------------------------------------------------
        # Full incoming additive message at every node:
        #
        #   m_u = sum_{w -> u} p_wu rho_{w->u}(t-1)
        #
        # This is also what is needed for the true node marginals.
        # ----------------------------------------------------------
        node_msg = np.bincount(
            dst,
            weights=edge_msg,
            minlength=graph.n
        ).astype(dtype, copy=False)

        # ----------------------------------------------------------
        # Node-level newly active mass.
        # ----------------------------------------------------------
        r = (1.0 - cumulative) * node_msg

        # ----------------------------------------------------------
        # Cavity message for every outgoing edge e = u -> v.
        #
        # Start with all influence reaching u...
        # ----------------------------------------------------------
        cavity_msg = node_msg[src].copy()

        # ...and remove the contribution v -> u.
        cavity_msg[has_rev] -= edge_msg[rev[has_rev]]

        # ----------------------------------------------------------
        # Newly active cavity mass:
        #
        # rho_{u->v}(t)
        #   = (1 - a_{u->v}(<t))
        #       * sum_{w->u, w != v} p_wu rho_{w->u}(t-1)
        # ----------------------------------------------------------
        rho_new = (1.0 - cavity_cumulative) * cavity_msg

        # Update cumulative states only after both updates were
        # computed from the previous time step.
        cumulative += r
        cavity_cumulative += rho_new

        rho = rho_new

    return np.clip(cumulative, 0.0, 1.0)

def swe(
    graph: ICGraph,
    prior_B,
    T: int,
    eps: float = 1e-15,
) -> np.ndarray:
    p0 = _prior_array(graph, prior_B, dtype=np.float64)

    if int(T) <= 0 or graph.m == 0:
        return p0.copy()

    src = graph.src
    dst = graph.dst.astype(np.int64, copy=False)
    p_e = graph.prob.astype(np.float64, copy=False)

    # B(0)
    B_prev = p0.copy()

    # B(-1), chosen so that
    # A(0) = B(0) - B(-1) = p0.
    B_prevprev = np.zeros_like(B_prev)

    for _ in range(int(T)):
        # Probability of becoming newly active in the previous step.
        A_prev = B_prev - B_prevprev

        vals = np.clip(
            p_e * A_prev[src],
            0.0,
            1.0 - eps,
        )

        log_prod = np.bincount(
            dst,
            weights=np.log1p(-vals),
            minlength=graph.n,
        )

        prod_term = np.exp(log_prod)

        B_new = 1.0 - (1.0 - B_prev) * prod_term
        np.clip(B_new, 0.0, 1.0, out=B_new)

        B_prevprev, B_prev = B_prev, B_new

    return B_prev


def swe_cavity(
    graph: ICGraph,
    prior_probs,
    T: int = 10,
    eps: float = 1e-15,
) -> np.ndarray:
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)

    if graph.m == 0:
        return p0

    src = graph.src
    p_e = graph.prob.astype(np.float64, copy=False)
    rev = graph.ensure_reverse()

    # Current node-level cumulative estimate B(t)
    B = p0.copy()

    # M(0) and M(-1)
    M_prev = p0[src].copy()
    M_prevprev = np.zeros_like(M_prev)

    mask = rev >= 0

    for _ in range(int(T)):
        # Newly activated cavity mass
        delta = M_prev - M_prevprev

        q = np.clip(
            1.0 - p_e * delta,
            eps,
            1.0
        )

        prod_full = _incoming_log_product(graph, q)

        # Node update: this is essential.
        B = 1.0 - (1.0 - B) * prod_full
        np.clip(B, 0.0, 1.0, out=B)

        # Cavity update
        prod_cavity = prod_full[src].copy()
        prod_cavity[mask] /= q[rev[mask]]
        np.clip(prod_cavity, 0.0, 1.0, out=prod_cavity)

        M_new = (
            M_prev
            + (1.0 - M_prev) * (1.0 - prod_cavity)
        )
        np.clip(M_new, 0.0, 1.0, out=M_new)

        M_prevprev, M_prev = M_prev, M_new

    return B


def swe_no_cavity(
    graph: ICGraph,
    prior_probs,
    T: int,
    a: float = 1.0,
    layers: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    dtype = np.result_type(graph.prob.dtype, np.float32)
    p0 = _prior_array(graph, prior_probs, dtype=dtype)
    if graph.m == 0:
        return p0

    src = graph.src
    p_e = graph.prob.astype(dtype, copy=False)
    rev = graph.ensure_reverse()

    m_e = p0[src].copy()
    survival_e = np.zeros(graph.m, dtype=dtype)
    node_cum = p0.copy()

    for _ in range(int(T)):
        active_by_now_e = 1.0 - (1.0 - survival_e) * (1.0 - m_e)
        q_e = np.clip(1.0 - p_e * m_e, eps, 1.0)
        prod_incoming = _incoming_log_product(graph, q_e).astype(dtype, copy=False)

        prod_excl = prod_incoming[src].copy()
        mask = rev >= 0
        prod_excl[mask] /= q_e[rev[mask]]
        prod_excl = np.clip(prod_excl, 0.0, 1.0)

        hazard_e = np.clip(1.0 - prod_excl, 0.0, 1.0)
        new_m_e = np.clip((1.0 - active_by_now_e) * hazard_e, 0.0, 1.0)
        survival_e, m_e = active_by_now_e, new_m_e

        node_hazard = np.clip(1.0 - prod_incoming, 0.0, 1.0)
        node_new = np.clip((1.0 - node_cum) * node_hazard, 0.0, 1.0)
        node_cum = 1.0 - (1.0 - node_cum) * (1.0 - node_new)
        np.clip(node_cum, 0.0, 1.0, out=node_cum)

    probs = node_cum

    for _ in range(int(layers)):
        vals = np.clip(p_e * probs[src], 0.0, 1.0 - eps)
        log_prod = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=np.log1p(-vals),
            minlength=graph.n,
        )
        neighbor_term = np.exp(log_prod)
        probs = 1.0 - (1.0 - p0) * neighbor_term
        np.clip(probs, 0.0, 1.0, out=probs)

    return np.clip(probs, 0.0, 1.0)


# -----------------------------------------------------------------------------
# SPM / SP1M
# -----------------------------------------------------------------------------


def SPM(
    graph: ICGraph,
    prior_probs,
    eps: float = 1e-12,
    seed_eps: float = 0.0,
) -> np.ndarray:
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)
    sources = np.flatnonzero(p0 > seed_eps)
    if sources.size == 0:
        return np.zeros(graph.n, dtype=np.float64)

    dist = _directed_multi_source_bfs_dist(graph, sources)
    reachable = dist >= 0
    if not np.any(reachable):
        return np.zeros(graph.n, dtype=np.float64)

    max_d = int(dist[reachable].max())
    P = np.zeros((max_d + 1, graph.n), dtype=np.float64)
    P[0] = p0
    p_e = graph.prob.astype(np.float64, copy=False)

    for t in range(1, max_d + 1):
        x_e = np.clip(p_e * P[t - 1, graph.src], 0.0, 1.0 - eps)
        log_prod = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=np.log1p(-x_e),
            minlength=graph.n,
        )
        P[t] = np.clip(1.0 - np.exp(log_prod), 0.0, 1.0)

    out = np.zeros(graph.n, dtype=np.float64)
    ar = np.arange(graph.n)
    out[reachable] = P[dist[reachable], ar[reachable]]
    return np.clip(out, 0.0, 1.0)


def SP1M(
    graph: ICGraph,
    prior_probs,
    eps: float = 1e-12,
    seed_eps: float = 0.0,
) -> np.ndarray:
    p0 = _prior_array(graph, prior_probs, dtype=np.float64)
    sources = np.flatnonzero(p0 > seed_eps)
    if sources.size == 0:
        return np.zeros(graph.n, dtype=np.float64)

    dist = _directed_multi_source_bfs_dist(graph, sources)
    reachable = dist >= 0
    if not np.any(reachable):
        return np.zeros(graph.n, dtype=np.float64)

    max_d1 = int(dist[reachable].max()) + 1
    P = np.zeros((max_d1 + 1, graph.n), dtype=np.float64)
    P[0] = p0
    p_e = graph.prob.astype(np.float64, copy=False)

    for t in range(1, max_d1 + 1):
        x_e = np.clip(p_e * P[t - 1, graph.src], 0.0, 1.0 - eps)
        log_prod = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=np.log1p(-x_e),
            minlength=graph.n,
        )
        base = np.clip(1.0 - np.exp(log_prod), 0.0, 1.0)
        P[t] = np.clip((1.0 - P[t - 1]) * base, 0.0, 1.0)

    out = np.zeros(graph.n, dtype=np.float64)
    ar = np.arange(graph.n)
    out[reachable] = (
        P[dist[reachable], ar[reachable]]
        + P[dist[reachable] + 1, ar[reachable]]
    )
    return np.clip(out, 0.0, 1.0)


# -----------------------------------------------------------------------------
# PageRank baseline
# -----------------------------------------------------------------------------


def pagerank(
    graph: ICGraph,
    prior_probs=None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-12,
    transition: str = "degree",
    eps: float = 1e-20,
) -> np.ndarray:
    n = graph.n
    if n == 0:
        return np.array([], dtype=np.float64)

    if prior_probs is None:
        v = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        v = _prior_array(graph, prior_probs, dtype=np.float64)
        s = v.sum()
        v = v / s if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    out_degree = np.diff(graph.out_indptr).astype(np.float64)

    if transition == "degree":
        out_w = out_degree
        edge_weight = np.ones(graph.m, dtype=np.float64)
    elif transition == "edge_probs":
        edge_weight = graph.prob.astype(np.float64, copy=False)
        out_w = np.bincount(
            graph.src.astype(np.int64, copy=False),
            weights=edge_weight,
            minlength=n,
        )
    else:
        raise ValueError('transition must be "degree" or "edge_probs".')

    dangling = out_w <= eps
    inv_out = np.zeros(n, dtype=np.float64)
    inv_out[~dangling] = 1.0 / out_w[~dangling]
    transition_e = edge_weight * inv_out[graph.src]

    r = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(int(max_iter)):
        linked = np.bincount(
            graph.dst.astype(np.int64, copy=False),
            weights=transition_e * r[graph.src],
            minlength=n,
        )
        dmass = r[dangling].sum()
        r_new = alpha * (linked + dmass * v) + (1.0 - alpha) * v
        s = r_new.sum()
        if s > 0:
            r_new /= s
        if np.linalg.norm(r_new - r, ord=1) <= tol:
            r = r_new
            break
        r = r_new

    return r


# -----------------------------------------------------------------------------
# Triangle-selective r=2 DMP
# -----------------------------------------------------------------------------


def dmp_est_r2(
    graph: ICGraph,
    prior_probs,
    T: int,
    eps: float = 1e-20,
) -> np.ndarray:
    """Triangle-selective second-order cavity DMP on ICGraph.

    The recurrence is the same as the uploaded implementation.  The refactor
    removes the O(m) Python `(src,dst)->edge` dictionary and instead exploits
    sorted adjacency plus binary-search edge lookup.
    """
    dtype = np.float64
    p0 = _prior_array(graph, prior_probs, dtype=dtype)
    if graph.m == 0:
        return p0
    if np.any(graph.src == graph.dst):
        raise ValueError("dmp_est_r2 does not support self-loops.")

    src, dst = graph.src, graph.dst
    p_e = graph.prob.astype(dtype, copy=False)
    rev = graph.ensure_reverse()
    in_indptr, in_edges = graph.ensure_incoming()

    triangle_prev, triangle_next, triangle_rotate = (
        graph.ensure_r2_triangles()
    )
    
    num_triangle_states = len(triangle_prev)

    m1 = p0[src].copy()
    m2 = p0[src[triangle_prev]].copy() if num_triangle_states else np.empty(0, dtype=dtype)

    for _ in range(int(T)):
        q1 = np.clip(1.0 - p_e * m1, eps, 1.0)
        log_q1 = np.log(q1)
        sum_log_incoming = np.bincount(
            dst.astype(np.int64, copy=False),
            weights=log_q1,
            minlength=graph.n,
        )

        log_product_for_edge = sum_log_incoming[src].copy()
        has_reverse = rev >= 0
        log_product_for_edge[has_reverse] -= log_q1[rev[has_reverse]]

        if num_triangle_states:
            q2 = np.clip(1.0 - p_e[triangle_prev] * m2, eps, 1.0)
            log_q2 = np.log(q2)
            correction = np.bincount(
                triangle_next.astype(np.int64, copy=False),
                weights=log_q2 - log_q1[triangle_prev],
                minlength=graph.m,
            )
            log_product_for_edge += correction

        log_product_for_edge = np.minimum(log_product_for_edge, 0.0)
        product_for_edge = np.exp(log_product_for_edge)
        new_m1 = np.clip(
            1.0 - (1.0 - p0[src]) * product_for_edge,
            0.0,
            1.0,
        )

        if num_triangle_states:
            log_product_for_m2 = (
                log_product_for_edge[triangle_prev]
                - log_q2[triangle_rotate]
            )
            log_product_for_m2 = np.minimum(log_product_for_m2, 0.0)
            product_for_m2 = np.exp(log_product_for_m2)
            m2 = np.clip(
                1.0 - (1.0 - p0[src[triangle_prev]]) * product_for_m2,
                0.0,
                1.0,
            )

        m1 = new_m1

    q1 = np.clip(1.0 - p_e * m1, eps, 1.0)
    sum_log_incoming = np.bincount(
        dst.astype(np.int64, copy=False),
        weights=np.log(q1),
        minlength=graph.n,
    )
    product_incoming = np.exp(sum_log_incoming)
    pi = 1.0 - (1.0 - p0) * product_incoming
    return np.clip(pi, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Hyperparameter helper
# -----------------------------------------------------------------------------


def get_best_parameters(
    graph: ICGraph,
    prior_probs,
    true_probs,
    method: str = "swe_no",
    eps: float = 1e-12,
    max_t: int = 10,
    min_t: int = 1,
    max_layers: int = 10,
):
    true_probs = np.asarray(true_probs, dtype=float)

    def rmse_fn(probs):
        probs = np.asarray(probs, dtype=float)
        return float(np.sqrt(np.mean((probs - true_probs) ** 2)))

    if method in {"swe_no_cavity", "swe_no"}:
        fn = swe_no if method == "swe_no" else swe_no_cavity
        best_rmse = np.inf
        best_t = min_t
        best_l = 0
        for t in range(min_t, max_t + 1):
            for l in range(max_layers + 1):
                probs = fn(graph, prior_probs, t, a=1.0, layers=l, eps=eps)
                score = rmse_fn(probs)
                if score < best_rmse:
                    best_rmse, best_t, best_l = score, t, l
        return best_t, best_l

    methods = {
        "swe": lambda t: swe(graph, prior_probs, t),
        "swe_cavity": lambda t: swe_cavity(graph, prior_probs, t),
        "additive_swe": lambda t: additive_swe(graph, prior_probs, t),
        "additive_swe_cavity": lambda t: additive_swe_cavity(graph, prior_probs, t),
        "Naive": lambda t: Naive(graph, prior_probs, t),
        "dmp_est": lambda t: dmp_est(graph, prior_probs, t),
        "dmp_est_r2": lambda t: dmp_est_r2(graph, prior_probs, t),
        "modified_ALE": lambda t: modified_ALE(graph, prior_probs, t),
        "modified_ALE2": lambda t: modified_ALE2(graph, prior_probs, t),
        "ALE_heuristic": lambda t: ALE_heuristic(graph, prior_probs, t),
        "ALE2": lambda t: ALE2(graph, prior_probs, t),
        "ALE2_heuristic": lambda t: ALE2(graph, prior_probs, t),
        "cavity_ALE": lambda t: cavity_ALE(graph, prior_probs, t),
        "modified_ALE_cavity": lambda t: modified_ALE_cavity(graph, prior_probs, t),
        "swe_hib_cavity": lambda t: swe_hib_cavity(graph, prior_probs, t, eps=1e-20),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}")

    best_rmse = np.inf
    best_t = min_t
    fn = methods[method]
    for t in range(min_t, max_t + 1):
        score = rmse_fn(fn(t))
        if score < best_rmse:
            best_rmse, best_t = score, t
    return best_t

"""Additional Independent Cascade approximations for the ICGraph setup.

Expected graph interface
------------------------
graph.n    : number of nodes
graph.src  : int array of source nodes, shape (m,)
graph.dst  : int array of destination nodes, shape (m,)
graph.prob : float array of IC edge probabilities, shape (m,)

All public functions return an ndarray of shape (graph.n,), aligned with node
indices 0, ..., graph.n-1.

Implemented methods
-------------------
sss            : SteadyStateSpread fixed-point approximation.
sss_noself     : Yang-Brenner-Giua SSS-Noself, generalized to probabilistic
                 initial activations. Faithful but intentionally expensive.
mia            : Maximum Influence Arborescence marginal approximation, with
                 a path-probability threshold theta.
burkholz       : Marginal output associated with Burkholz-Quackenbush TDA.
                 For unconditional node marginals this reduces to the BP/DMP-inf
                 marginals used by TDA; see the docstring of burkholz().

The original papers formulate SSS/SSS-Noself/MIA for deterministic seed sets.
Here they are extended to independent initial activation probabilities p0[v] by
replacing the deterministic non-seed survival factor with (1-p0[v]).
"""

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _arrays(graph, prior_probs):
    n = int(graph.n)
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    prob = np.asarray(graph.prob, dtype=np.float64)
    prior = np.asarray(prior_probs, dtype=np.float64)

    if src.ndim != 1 or dst.ndim != 1 or prob.ndim != 1:
        raise ValueError("graph.src, graph.dst and graph.prob must be 1-D arrays")
    if not (len(src) == len(dst) == len(prob)):
        raise ValueError("graph.src, graph.dst and graph.prob must have equal length")
    if prior.shape != (n,):
        raise ValueError(f"prior_probs must have shape ({n},), got {prior.shape}")
    if len(src):
        if src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
            raise ValueError("graph edge endpoints must lie in [0, graph.n)")
    if np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("graph.prob must lie in [0, 1]")
    if np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("prior_probs must lie in [0, 1]")

    return n, src, dst, prob, prior


def _incoming_index(n: int, dst: np.ndarray):
    """Return (edge_order, indptr) grouping edge indices by destination."""
    order = np.argsort(dst, kind="stable")
    counts = np.bincount(dst, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return order, indptr


def _log1m(x: np.ndarray, eps: float) -> np.ndarray:
    """Stable log(1-x), clipping only to avoid log(0) / inf-inf arithmetic."""
    return np.log1p(-np.clip(x, 0.0, 1.0 - eps))


def _sss_step(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    prob: np.ndarray,
    prior: np.ndarray,
    state: np.ndarray,
    eps: float,
) -> np.ndarray:
    if len(src) == 0:
        return prior.copy()
    log_terms = _log1m(prob * state[src], eps)
    incoming_log = np.bincount(dst, weights=log_terms, minlength=n)
    out = 1.0 - (1.0 - prior) * np.exp(incoming_log)
    return np.clip(out, 0.0, 1.0)


def _reverse_edge_index(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    """Find the reverse directed edge for every edge, or -1 if absent.

    Assumes a simple directed graph for exact one-to-one reverse matching. If
    parallel edges exist, the first matching reverse edge is used.
    """
    m = len(src)
    if m == 0:
        return np.empty(0, dtype=np.int64)

    key = src.astype(np.int64) * np.int64(n) + dst.astype(np.int64)
    rev_key = dst.astype(np.int64) * np.int64(n) + src.astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    pos = np.searchsorted(skey, rev_key)

    rev = np.full(m, -1, dtype=np.int64)
    valid = pos < m
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx):
        p = pos[valid_idx]
        hit = skey[p] == rev_key[valid_idx]
        ii = valid_idx[hit]
        rev[ii] = order[pos[ii]]
    return rev

# ---------------------------------------------------------------------------
# 1. SteadyStateSpread
# ---------------------------------------------------------------------------

def sss(
    graph,
    prior_probs,
    tol: float = 1e-8,
    max_iter: int = 1000,
    eps: float = 1e-12,
) -> np.ndarray:
    """SteadyStateSpread (SSS).

    Generalized from a deterministic seed set to independent initial
    activation probabilities. The fixed-point equation is

        p[v] = 1 - (1-p0[v]) * prod_{u->v}(1 - w[u,v] p[u]).

    Parameters
    ----------
    graph : ICGraph-like
        Must expose n, src, dst, prob.
    prior_probs : array-like, shape (n,)
        Independent initial activation probabilities.
    tol : float
        L-infinity fixed-point tolerance.
    max_iter : int
        Maximum number of synchronous fixed-point iterations.
    eps : float
        Numerical clipping used only inside log(1-x).
    """
    n, src, dst, prob, prior = _arrays(graph, prior_probs)
    state = prior.copy()

    for _ in range(int(max_iter)):
        new = _sss_step(n, src, dst, prob, prior, state, eps)
        if np.max(np.abs(new - state)) <= tol:
            return new
        state = new

    return state


# ---------------------------------------------------------------------------
# 2. SSS-Noself
# ---------------------------------------------------------------------------

def sss_noself(
    graph,
    prior_probs,
    tol: float = 1e-8,
    max_iter: int = 1000,
    eps: float = 1e-12,
) -> np.ndarray:
    """SSS-Noself of Yang, Brenner & Giua (2018).

    This is the faithful expensive construction, not an edge-cavity shortcut.
    For each target q it solves SSS in the network where q is forced inactive
    (equivalent here to removing q's incident influence), then uses those cavity
    probabilities to update q itself.

    The paper assumes deterministic seeds. For probabilistic p0[q], this uses

        p[q] = 1 - (1-p0[q]) * prod_{u->q}(1 - w[u,q] p^{[q]}[u]),

    where p^{[q]} is the SSS fixed point with q forced inactive.

    Warning
    -------
    This requires one graph-wide fixed-point solve per target node. It is thus
    unsuitable for the large-graph benchmark except possibly on small subsets;
    that unpleasant fact belongs to the method, not to NumPy's moral character.
    """
    n, src, dst, prob, prior = _arrays(graph, prior_probs)
    result = prior.copy()
    if n == 0 or len(src) == 0:
        return result

    in_order, in_indptr = _incoming_index(n, dst)

    for q in range(n):
        if prior[q] >= 1.0 - eps:
            result[q] = 1.0
            continue

        lo, hi = in_indptr[q], in_indptr[q + 1]
        if lo == hi:
            result[q] = prior[q]
            continue

        cavity_prior = prior.copy()
        cavity_prior[q] = 0.0
        state = cavity_prior.copy()

        for _ in range(int(max_iter)):
            new = _sss_step(n, src, dst, prob, cavity_prior, state, eps)
            # q is removed/forced inactive in G[q].
            new[q] = 0.0
            if np.max(np.abs(new - state)) <= tol:
                state = new
                break
            state = new

        eidx = in_order[lo:hi]
        log_survival = np.sum(_log1m(prob[eidx] * state[src[eidx]], eps))
        result[q] = 1.0 - (1.0 - prior[q]) * np.exp(log_survival)

    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Maximum Influence Arborescence (MIA)
# ---------------------------------------------------------------------------

def mia(
    graph,
    prior_probs,
    theta: float = 0.01,
    eps: float = 1e-15,
) -> np.ndarray:
    """Maximum Influence Arborescence (MIA) marginal approximation.

    For each target v, construct MIIA(v, theta): the union of one maximum-
    probability path u -> ... -> v for every u whose path probability is at
    least theta. On that in-arborescence, compute the usual IC activation
    recursion exactly.

    Maximum-product paths are obtained by Dijkstra on the reversed graph with
    edge cost -log(w). The threshold stops the search once path probability
    drops below theta.

    The original MIA paper uses deterministic seeds. Here each node independently
    starts active with prior_probs[u], so the recursion becomes

        ap[u] = 1 - (1-p0[u]) prod_{x in N_in^MIIA(u)}(1-w[x,u] ap[x]).

    Parameters
    ----------
    theta : float in [0,1]
        Minimum retained maximum-path probability. theta=0 removes pruning and
        is usually a spectacularly bad runtime decision on large graphs.
    """
    n, src, dst, prob, prior = _arrays(graph, prior_probs)
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must lie in [0, 1]")
    if n == 0 or len(src) == 0:
        return prior.copy()

    in_order, in_indptr = _incoming_index(n, dst)
    edge_cost = np.full(len(prob), np.inf, dtype=np.float64)
    positive = prob > 0.0
    edge_cost[positive] = -np.log(prob[positive])
    cutoff = np.inf if theta == 0.0 else -np.log(max(theta, eps))

    out = np.empty(n, dtype=np.float64)
    INF = float("inf")

    for target in range(n):
        # Sparse local Dijkstra state: with a useful theta, MIIA should be local.
        dist = {target: 0.0}
        parent_edge = {}  # u -> edge (u, parent[u]) toward target
        heap = [(0.0, target)]

        while heap:
            d, x = heapq.heappop(heap)
            if d != dist.get(x, INF):
                continue
            if d > cutoff:
                break

            lo, hi = in_indptr[x], in_indptr[x + 1]
            for pos in range(lo, hi):
                e = int(in_order[pos])
                if not positive[e]:
                    continue
                u = int(src[e])
                nd = d + float(edge_cost[e])
                if nd > cutoff + 1e-15:
                    continue
                old = dist.get(u, INF)
                if nd + 1e-15 < old:
                    dist[u] = nd
                    parent_edge[u] = e
                    heapq.heappush(heap, (nd, u))

        # Evaluate the IC recursion on the resulting rooted in-arborescence.
        # Use a leaf-to-root topological swe_noep instead of distance sorting,
        # because probability-1 edges have zero Dijkstra cost and hence ties.
        survival = {u: float(1.0 - prior[u]) for u in dist}
        child_count = {u: 0 for u in dist}
        for u, e in parent_edge.items():
            p = int(dst[e])
            child_count[p] += 1

        queue = deque(u for u, c in child_count.items() if c == 0)
        processed = 0
        while queue:
            u = queue.popleft()
            processed += 1
            ap_u = 1.0 - survival[u]
            e = parent_edge.get(u)
            if e is None:
                continue  # root / target
            p = int(dst[e])
            survival[p] *= 1.0 - float(prob[e]) * ap_u
            child_count[p] -= 1
            if child_count[p] == 0:
                queue.append(p)

        if processed != len(dist):
            raise RuntimeError("MIA predecessor structure unexpectedly contains a cycle")

        out[target] = 1.0 - survival[target]

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Burkholz & Quackenbush TDA, marginal-output version
# ---------------------------------------------------------------------------

# Convenient aliases matching common paper naming.
SteadyStateSpread = sss
SSS_Noself = sss_noself
MIA = mia

__all__ = [
    "ICGraph",
    "optimized_independent_cascade",
    "dmp_est",
    "dmp_inf",
    "dmp_python",
    "dmp_est_r2",
    "dmp_um_python",
    "swe",
    "swe_cavity",
    "swe_no",
    "swe_no_cavity",
    "additive_swe",
    "additive_swe_cavity",
    "swe_hib_cavity",
    "ALE_heuristic",
    "ALE2",
    "ALE2_heuristic",
    "cavity_ALE",
    "modified_ALE",
    "modified_ALE2",
    "modified_ALE_cavity",
    "Naive",
    "SPM",
    "SP1M",
    "pagerank",
    "sss",
    "sss_noself",
    "mia",
    "SteadyStateSpread",
    "SSS_Noself",
    "MIA",
    "get_best_parameters",
]
