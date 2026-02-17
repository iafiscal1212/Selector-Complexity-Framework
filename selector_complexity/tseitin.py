"""
Tseitin Axiom Builders
======================

Generates Tseitin tautologies over arbitrary graphs.
Includes built-in expander graph constructors.

A Tseitin formula on graph G = (V, E) with charges chi: V -> {0, 1}
is unsatisfiable when sum(chi) is odd (parity argument).

Each vertex v with incident edges e_1, ..., e_k produces the axiom:
    prod_{i=1}^{k} (1 - 2*x_{e_i}) = (-1)^{chi(v)}

This is the multilinear XOR-parity constraint.

Graphs:
    petersen_graph     -- 10 vertices, 15 edges, 3-regular
    cube_graph         -- 8 vertices, 12 edges, 3-regular
    circulant_graph    -- n vertices with given generators
    random_regular     -- random d-regular graph on n vertices

Author: Carmen Esteban
License: MIT
"""

import random


# =====================================================================
# GRAPH CONSTRUCTORS
# =====================================================================

def petersen_graph():
    """Petersen graph: 10 vertices, 15 edges, 3-regular.

    Classic small expander with spectral gap ~0.38.

    Returns
    -------
    edges : list of (int, int)
    num_vertices : int
    """
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return outer + inner + spokes, 10


def cube_graph():
    """Cube graph Q3: 8 vertices, 12 edges, 3-regular.

    Hypercube in 3 dimensions.

    Returns
    -------
    edges : list of (int, int)
    num_vertices : int
    """
    edges = []
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j))
    return edges, 8


def circulant_graph(n, generators):
    """Circulant graph C_n(g_1, g_2, ...).

    Vertices are 0, ..., n-1. Vertex i connects to i +/- g_j (mod n).

    Parameters
    ----------
    n : int
        Number of vertices.
    generators : list of int
        Connection offsets (e.g., [1, 3] for C_n(1,3)).

    Returns
    -------
    edges : list of (int, int)
    num_vertices : int
    """
    edge_set = set()
    for i in range(n):
        for g in generators:
            j = (i + g) % n
            edge = (min(i, j), max(i, j))
            if edge[0] != edge[1]:
                edge_set.add(edge)
    return sorted(edge_set), n


def random_regular(n, d, seed=42):
    """Random d-regular graph on n vertices (pairing model).

    Uses a simple retry-based construction. Requires n*d to be even.

    Parameters
    ----------
    n : int
        Number of vertices.
    d : int
        Degree of each vertex.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    edges : list of (int, int)
    num_vertices : int
    """
    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even")

    rng = random.Random(seed)

    for _ in range(100):  # retry attempts
        stubs = []
        for v in range(n):
            stubs.extend([v] * d)
        rng.shuffle(stubs)

        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                ok = False
                break
            edges.add(edge)

        if ok and len(edges) == n * d // 2:
            return sorted(edges), n

    raise RuntimeError("Failed to generate random regular graph after 100 attempts")


# =====================================================================
# TSEITIN AXIOM BUILDER
# =====================================================================

def tseitin_axioms(edges, num_vertices, charges=None):
    """Generate Tseitin axioms on a graph.

    Each vertex v with incident edges e_1, ..., e_k and charge c_v
    produces the polynomial axiom:

        prod_{i=1}^{k} (1 - 2*x_{e_i}) - (-1)^{c_v} = 0

    The system is unsatisfiable when sum(charges) is odd.

    Parameters
    ----------
    edges : list of (int, int)
        Graph edges (0-indexed vertices).
    num_vertices : int
        Number of vertices.
    charges : list of int, optional
        Charge per vertex (0 or 1). Must have odd total for
        unsatisfiability. Default: vertex 0 gets charge 1, rest 0.

    Returns
    -------
    axioms : list of list of (coef, frozenset) tuples
        Polynomial axioms in standard framework format.
    num_vars : int
        Number of variables (= number of edges).
    var_x : dict
        Mapping (u, v) -> variable index for each edge.
    """
    if charges is None:
        charges = [0] * num_vertices
        charges[0] = 1

    # One variable per edge
    var_x = {}
    for i, (u, v) in enumerate(edges):
        var_x[(u, v)] = i
        var_x[(v, u)] = i
    num_vars = len(edges)

    axioms = []

    for v in range(num_vertices):
        # Edges incident to v
        incident = set()
        for u, w in edges:
            if u == v:
                incident.add(var_x[(u, w)])
            elif w == v:
                incident.add(var_x[(u, w)])
        incident = sorted(incident)

        if not incident:
            continue

        # Expand product: prod(1 - 2*x_i)
        # = sum over subsets S: (-2)^|S| * prod_{i in S} x_i
        terms = {}
        for mask in range(1 << len(incident)):
            subset = []
            for bit in range(len(incident)):
                if mask & (1 << bit):
                    subset.append(incident[bit])
            coef = (-2.0) ** len(subset)
            mono = frozenset(subset)
            terms[mono] = terms.get(mono, 0.0) + coef

        # Subtract (-1)^charge
        sign = (-1.0) ** charges[v]
        terms[frozenset()] = terms.get(frozenset(), 0.0) - sign

        axiom = [(c, m) for m, c in terms.items() if abs(c) > 1e-15]
        if axiom:
            axioms.append(axiom)

    return axioms, num_vars, var_x
