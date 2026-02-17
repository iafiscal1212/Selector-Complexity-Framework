"""
IPS Solvers
===========

Builds sparse IPS matrices and solves via LSQR.

Functions:
    build_matrix             -- original frozenset-based matrix builder
    find_certificate         -- single-degree certificate search
    build_matrix_tuples      -- optimized tuple-based matrix builder (~2x faster hashing)
    find_certificate_blocked -- block-decomposed certificate search
    IncrementalIPSState      -- incremental matrix builder across degrees
    incremental_certificate_search -- search reusing state across degrees

Author: Carmen Esteban
License: MIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from itertools import combinations


# =====================================================================
# ORIGINAL API (preserved for backwards compatibility)
# =====================================================================

def build_matrix(axioms, num_vars, d_max):
    all_monoms = []
    monom_to_idx = {}
    for d in range(d_max + 1):
        for combo in combinations(range(num_vars), d):
            m = frozenset(combo)
            monom_to_idx[m] = len(all_monoms)
            all_monoms.append(m)
    num_monoms = len(all_monoms)
    rows, cols, vals = [], [], []
    total_unknowns = 0
    for ax in axioms:
        deg_ax = max(len(m) for c, m in ax)
        deg_mult = max(0, d_max - deg_ax)
        for d in range(deg_mult + 1):
            for combo in combinations(range(num_vars), d):
                m_mult = frozenset(combo)
                col = total_unknowns
                total_unknowns += 1
                for coef_ax, m_ax in ax:
                    m_prod = m_mult | m_ax
                    if len(m_prod) <= d_max and m_prod in monom_to_idx:
                        rows.append(monom_to_idx[m_prod])
                        cols.append(col)
                        vals.append(coef_ax)
    if total_unknowns == 0:
        return sparse.csr_matrix((num_monoms, 0)), np.zeros(num_monoms), num_monoms, 0
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(num_monoms, total_unknowns))
    b = np.zeros(num_monoms)
    b[monom_to_idx[frozenset()]] = 1.0
    return A, b, num_monoms, total_unknowns


def find_certificate(axioms, num_vars, d_max, atol=1e-12):
    A, b, nm, nu = build_matrix(axioms, num_vars, d_max)
    if nu == 0:
        return None
    res = lsqr(A, b, atol=atol, btol=atol, iter_lim=10000)
    x = res[0]
    residual = np.linalg.norm(A @ x - b)
    if residual < 1e-6:
        size = int(np.sum(np.abs(x) > 1e-8))
        return {"degree": d_max, "size": size, "num_monoms": nm,
                "num_unknowns": nu, "residual": float(residual), "coefficients": x}
    return None


# =====================================================================
# OPT 2: TUPLE-BASED MONOMIAL INDEXING (~2x faster hashing)
# =====================================================================

def _tuple_union(a, b):
    """Merge two sorted tuples into a sorted tuple (set union)."""
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        elif a[i] > b[j]:
            result.append(b[j])
            j += 1
        else:
            result.append(a[i])
            i += 1
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return tuple(result)


def build_matrix_tuples(axioms, num_vars, d_max):
    """Build IPS matrix using tuple-based monomial indexing.

    Same semantics as build_matrix but uses tuple(sorted(combo))
    instead of frozenset for monomial keys. Tuple hashing is ~2x
    faster than frozenset hashing.

    Returns
    -------
    A : scipy.sparse.csr_matrix
    b : numpy.ndarray
    num_monoms : int
    total_unknowns : int
    """
    all_monoms = []
    monom_to_idx = {}
    for d in range(d_max + 1):
        for combo in combinations(range(num_vars), d):
            t = combo  # already sorted tuple from combinations
            monom_to_idx[t] = len(all_monoms)
            all_monoms.append(t)
    num_monoms = len(all_monoms)

    rows, cols, vals = [], [], []
    total_unknowns = 0

    # Pre-convert axiom monomials to sorted tuples
    axioms_t = []
    for ax in axioms:
        ax_t = [(c, tuple(sorted(m))) for c, m in ax]
        axioms_t.append(ax_t)

    for ax_t in axioms_t:
        deg_ax = max(len(m) for _, m in ax_t)
        deg_mult = max(0, d_max - deg_ax)

        for d in range(deg_mult + 1):
            for combo in combinations(range(num_vars), d):
                m_mult = combo
                col = total_unknowns
                total_unknowns += 1

                for coef_ax, m_ax in ax_t:
                    m_prod = _tuple_union(m_mult, m_ax)
                    if len(m_prod) <= d_max and m_prod in monom_to_idx:
                        rows.append(monom_to_idx[m_prod])
                        cols.append(col)
                        vals.append(coef_ax)

    if total_unknowns == 0:
        return sparse.csr_matrix((num_monoms, 0)), np.zeros(num_monoms), num_monoms, 0

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(num_monoms, total_unknowns))
    b = np.zeros(num_monoms)
    b[monom_to_idx[()]] = 1.0
    return A, b, num_monoms, total_unknowns


# =====================================================================
# OPT 2: BLOCK DECOMPOSITION VIA UNION-FIND
# =====================================================================

class _UnionFind:
    """Simple Union-Find for variable block detection."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _find_independent_blocks(axioms, num_vars):
    """Find independent variable blocks using Union-Find.

    Two variables are in the same block if they appear in the same axiom.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    list of dict
        Each dict has 'vars' (set of var indices), 'axioms' (list of axiom indices).
        Returns None if there's only one block (no decomposition benefit).
    """
    uf = _UnionFind(num_vars)

    # Union variables that appear in the same axiom
    axiom_vars = []
    for ax in axioms:
        ax_vs = set()
        for _, m in ax:
            ax_vs.update(m)
        ax_vs = sorted(ax_vs)
        axiom_vars.append(ax_vs)
        for i in range(1, len(ax_vs)):
            uf.union(ax_vs[0], ax_vs[i])

    # Group by root
    block_map = {}
    for v in range(num_vars):
        root = uf.find(v)
        if root not in block_map:
            block_map[root] = set()
        block_map[root].add(v)

    if len(block_map) <= 1:
        return None  # Single block, no decomposition benefit

    # Assign axioms to blocks
    blocks = []
    for root, var_set in block_map.items():
        block_axiom_indices = []
        for i, ax_vs in enumerate(axiom_vars):
            if ax_vs and uf.find(ax_vs[0]) == root:
                block_axiom_indices.append(i)
        blocks.append({
            'vars': var_set,
            'axioms': block_axiom_indices,
        })

    return blocks


def _remap_block(axioms, block_vars):
    """Remap axioms to use local variable indices within a block.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        Original axioms (only those belonging to this block).
    block_vars : set of int
        Variable indices in this block.

    Returns
    -------
    remapped_axioms : list
    local_num_vars : int
    """
    sorted_vars = sorted(block_vars)
    var_map = {v: i for i, v in enumerate(sorted_vars)}
    local_num_vars = len(sorted_vars)

    remapped = []
    for ax in axioms:
        new_ax = []
        for coef, mono in ax:
            new_mono = frozenset(var_map[v] for v in mono if v in var_map)
            new_ax.append((coef, new_mono))
        remapped.append(new_ax)

    return remapped, local_num_vars


def find_certificate_blocked(axioms, num_vars, d_max, atol=1e-12):
    """Find IPS certificate using block decomposition when possible.

    If the constraint graph is disconnected, solves each block independently.
    This can be dramatically faster: C(n/k, d) << C(n, d).

    Falls back to standard find_certificate if only one block exists.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int
    d_max : int
    atol : float

    Returns
    -------
    dict or None
        Same format as find_certificate, with additional 'blocks' field.
    """
    blocks = _find_independent_blocks(axioms, num_vars)

    if blocks is None:
        # Single block: use standard solver
        result = find_certificate(axioms, num_vars, d_max, atol)
        if result:
            result['blocks'] = 1
        return result

    # Multi-block: solve each independently
    total_size = 0
    total_monoms = 0
    total_unknowns = 0
    max_residual = 0.0
    all_found = True

    for block in blocks:
        block_axioms = [axioms[i] for i in block['axioms']]
        if not block_axioms:
            continue

        remapped, local_nv = _remap_block(block_axioms, block['vars'])
        cert = find_certificate(remapped, local_nv, d_max, atol)

        if cert is None:
            all_found = False
            break

        total_size += cert['size']
        total_monoms += cert['num_monoms']
        total_unknowns += cert['num_unknowns']
        max_residual = max(max_residual, cert['residual'])

    if all_found:
        return {
            'degree': d_max,
            'size': total_size,
            'num_monoms': total_monoms,
            'num_unknowns': total_unknowns,
            'residual': max_residual,
            'blocks': len(blocks),
        }

    return None


# =====================================================================
# OPT 3: INCREMENTAL IPS STATE
# =====================================================================

class IncrementalIPSState:
    """Maintains incremental state for IPS matrix construction across degrees.

    Instead of rebuilding the entire matrix from scratch for each degree d,
    this class accumulates COO entries and only processes new monomials and
    multipliers when extending to a higher degree.

    When extending, old multiplier columns may gain new entries: products that
    exceeded old_degree but fit within d_new. These are re-scanned correctly.

    Usage
    -----
    state = IncrementalIPSState(axioms, num_vars)
    for d in range(2, max_degree + 1):
        A, b, nm, nu = state.extend_to_degree(d)
        # solve with A, b ...
    """

    def __init__(self, axioms, num_vars):
        self.axioms = axioms
        self.num_vars = num_vars
        self.current_degree = -1

        # Accumulated COO entries
        self._rows = []
        self._cols = []
        self._vals = []
        self._total_unknowns = 0

        # Monomial index (tuple-based for speed)
        self._all_monoms = []
        self._monom_to_idx = {}

        # Track what's been processed per axiom
        # For each axiom index, the max multiplier degree already processed
        self._axiom_max_mult_deg = {}

        # Track column assignments: list of (axiom_idx, multiplier_tuple)
        self._column_info = []

        # Pre-convert axiom monomials to tuples
        self._axioms_t = []
        self._axiom_degrees = []
        for ax in axioms:
            ax_t = [(c, tuple(sorted(m))) for c, m in ax]
            self._axioms_t.append(ax_t)
            deg = max(len(m) for _, m in ax_t)
            self._axiom_degrees.append(deg)

    def extend_to_degree(self, d_new):
        """Extend the IPS matrix to degree d_new.

        Enumerates new monomials, adds new multiplier columns, and
        re-scans old columns for products that now fit within d_new.

        Parameters
        ----------
        d_new : int
            New maximum degree. Must be >= current_degree.

        Returns
        -------
        A : scipy.sparse.csr_matrix
        b : numpy.ndarray
        num_monoms : int
        total_unknowns : int
        """
        if d_new <= self.current_degree:
            return self._build_current()

        old_degree = self.current_degree

        # Add new monomials for degrees (old_degree+1) to d_new
        for d in range(max(0, old_degree + 1), d_new + 1):
            for combo in combinations(range(self.num_vars), d):
                t = combo
                if t not in self._monom_to_idx:
                    self._monom_to_idx[t] = len(self._all_monoms)
                    self._all_monoms.append(t)

        # Re-scan old columns for entries that now fit in the expanded
        # degree range. This handles the case where a multiplier of degree
        # dm with an axiom term of degree deg_m produced a product of
        # degree > old_degree but <= d_new.
        if old_degree >= 0:
            for col_idx, (ai, m_mult) in enumerate(self._column_info):
                ax_t = self._axioms_t[ai]
                for coef_ax, m_ax in ax_t:
                    m_prod = _tuple_union(m_mult, m_ax)
                    prod_len = len(m_prod)
                    # Only add entries that were previously out of range
                    if (old_degree < prod_len <= d_new
                            and m_prod in self._monom_to_idx):
                        self._rows.append(self._monom_to_idx[m_prod])
                        self._cols.append(col_idx)
                        self._vals.append(coef_ax)

        # For each axiom, process new multiplier degrees
        for ax_idx, ax_t in enumerate(self._axioms_t):
            deg_ax = self._axiom_degrees[ax_idx]
            old_max_mult = self._axiom_max_mult_deg.get(ax_idx, -1)
            new_max_mult = max(0, d_new - deg_ax)

            if new_max_mult <= old_max_mult:
                continue

            # Only enumerate multipliers of degree > old_max_mult
            start_deg = old_max_mult + 1
            for dm in range(start_deg, new_max_mult + 1):
                for combo in combinations(range(self.num_vars), dm):
                    m_mult = combo
                    col = self._total_unknowns
                    self._total_unknowns += 1
                    self._column_info.append((ax_idx, m_mult))

                    for coef_ax, m_ax in ax_t:
                        m_prod = _tuple_union(m_mult, m_ax)
                        if len(m_prod) <= d_new and m_prod in self._monom_to_idx:
                            self._rows.append(self._monom_to_idx[m_prod])
                            self._cols.append(col)
                            self._vals.append(coef_ax)

            self._axiom_max_mult_deg[ax_idx] = new_max_mult

        self.current_degree = d_new
        return self._build_current()

    def _build_current(self):
        """Build CSR matrix from accumulated COO entries.

        Returns
        -------
        A, b, num_monoms, total_unknowns
        """
        num_monoms = len(self._all_monoms)

        if self._total_unknowns == 0:
            return (sparse.csr_matrix((num_monoms, 0)),
                    np.zeros(num_monoms), num_monoms, 0)

        A = sparse.csr_matrix(
            (self._vals, (self._rows, self._cols)),
            shape=(num_monoms, self._total_unknowns))
        b = np.zeros(num_monoms)
        b[self._monom_to_idx[()]] = 1.0
        return A, b, num_monoms, self._total_unknowns


def incremental_certificate_search(axioms, num_vars, max_degree=10,
                                    min_degree=2, atol=1e-12, verbose=False):
    """Search for IPS certificates using incremental matrix construction.

    Reuses monomial enumeration and COO entries across degrees,
    only adding new multiplier terms at each step.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int
    max_degree : int
    min_degree : int
    atol : float
    verbose : bool

    Returns
    -------
    list of dict
        Per-degree results (same format as _search_certificates).
    """
    from math import comb as math_comb
    import time

    state = IncrementalIPSState(axioms, num_vars)
    results = []

    for d in range(min_degree, max_degree + 1):
        num_monoms_est = sum(math_comb(num_vars, k) for k in range(d + 1))
        if num_monoms_est > 500000:
            if verbose:
                print("    d={}: ~{} monomials, skipping (too large)".format(
                    d, int(num_monoms_est)))
            break

        t0 = time.time()

        A, b, nm, nu = state.extend_to_degree(d)

        if nu == 0:
            elapsed = time.time() - t0
            results.append({
                'degree': d, 'feasible': False, 'residual': 1.0,
                'size': 0, 'num_monoms': nm,
                'num_unknowns': 0, 'time': elapsed,
            })
            if verbose:
                print("    d={}: no unknowns [{:.2f}s]".format(d, elapsed))
            continue

        sol = lsqr(A, b, atol=atol, btol=atol, iter_lim=10000)
        x = sol[0]
        residual = float(np.linalg.norm(A @ x - b))
        size = int(np.sum(np.abs(x) > 1e-8))
        elapsed = time.time() - t0

        feasible = residual < 1e-6
        results.append({
            'degree': d, 'feasible': feasible, 'residual': residual,
            'size': size, 'num_monoms': nm,
            'num_unknowns': nu, 'time': elapsed,
        })

        if verbose:
            status = "FEASIBLE" if feasible else "INFEASIBLE"
            print("    d={}: {} (res={:.2e}, size={}, monoms={}) [{:.2f}s]".format(
                d, status, residual, size, nm, elapsed))

        if feasible:
            break

    return results
