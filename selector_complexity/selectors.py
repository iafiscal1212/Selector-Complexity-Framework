import numpy as np
from itertools import combinations, permutations

def build_phpe_selectors(n):
    pigeons = list(range(1, n + 2))
    var_y = {}
    idx = 0
    for i, p in enumerate(pigeons):
        for p2 in pigeons[i + 1:]:
            var_y[(p, p2)] = idx
            idx += 1
    indicators = {}
    for p in pigeons:
        factors = []
        for q in pigeons:
            if q == p: continue
            if q < p:
                factors.append([(1.0, frozenset([var_y[(q, p)]]))])
            else:
                factors.append([(1.0, frozenset()), (-1.0, frozenset([var_y[(p, q)]]))])
        result = [(1.0, frozenset())]
        for factor in factors:
            new_result = []
            for c1, m1 in result:
                for c2, m2 in factor:
                    new_result.append((c1 * c2, m1 | m2))
            combined = {}
            for c, m in new_result:
                combined[m] = combined.get(m, 0) + c
            result = [(c, m) for m, c in combined.items() if abs(c) > 1e-15]
        indicators[p] = result
    return indicators, var_y

def test_s_only_feasibility(n, max_degree=3):
    pigeons = list(range(1, n + 2))
    var_s = {}
    idx = 0
    for p in pigeons:
        for q in pigeons:
            if p != q:
                var_s[(p, q)] = idx
                idx += 1
    num_s_vars = idx
    others = pigeons[1:]
    cycle_assignments = []
    for perm in permutations(others):
        cycle = (pigeons[0],) + perm
        assignment = {}
        for i in range(len(cycle)):
            p, q = cycle[i], cycle[(i + 1) % len(cycle)]
            for p2 in pigeons:
                for q2 in pigeons:
                    if p2 != q2:
                        assignment[var_s[(p2, q2)]] = 1 if (p2 == p and q2 == q) else 0
        cycle_assignments.append(assignment)
    monoms = [frozenset()]
    for d in range(1, max_degree + 1):
        for combo in combinations(range(num_s_vars), d):
            monoms.append(frozenset(combo))
    eval_matrix = np.zeros((len(cycle_assignments), len(monoms)))
    for i, assign in enumerate(cycle_assignments):
        for j, monom in enumerate(monoms):
            prod = 1.0
            for v in monom:
                prod *= assign.get(v, 0)
            eval_matrix[i, j] = prod
    unique_rows = np.unique(eval_matrix, axis=0)
    return len(unique_rows) == 1, len(cycle_assignments), len(unique_rows)


def enumerate_vc(n):
    """Enumerate valid configurations (placeholder for __init__ compatibility)."""
    raise NotImplementedError("enumerate_vc not yet implemented")


def build_phpc_explicit_selectors(n):
    """Build explicit selectors for PHP-C (placeholder for __init__ compatibility)."""
    raise NotImplementedError("build_phpc_explicit_selectors not yet implemented")
