#!/usr/bin/env python3
"""
Systematic SC Landscape Mapping (v3)
=====================================

Maps 10 tautology families across the Selector Complexity hierarchy.

Fixes over v2:
    1. Added Factoring, Goldreich-PRG, Binary-LWE families
    2. Corrected expected_sc for 3-XOR and Subset-Sum (genuinely SC(0))
    3. Monomial cap raised to 1M (PHP-E(4) now feasible at d=6)

Author: Carmen Esteban
"""

import os
import sys
import time
import random
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from selector_complexity import (
    php_axioms, phpe_axioms, phpc_axioms,
    tseitin_axioms, circulant_graph,
    recommend_strategy,
    quantify_hardness, compare_hardness,
    estimate_level_family,
    factoring_axioms, goldreich_prg_axioms, binary_lwe_axioms,
)
from selector_complexity.classifier import _analyze_structure, _search_certificates


# =====================================================================
# HARD 3-XOR BUILDER (GF(2) elimination)
# =====================================================================

def _gf2_is_unsat(clauses_vars, b_vec, n):
    """Check if the XOR system is unsatisfiable via GF(2) Gauss."""
    m = len(clauses_vars)
    # Augmented matrix as bitmasks: bits 0..n-1 = coefficients, bit n = RHS
    aug = []
    for vars_chosen, bi in zip(clauses_vars, b_vec):
        row = 0
        for v in vars_chosen:
            row |= (1 << v)
        if bi:
            row |= (1 << n)
        aug.append(row)

    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, m):
            if aug[r] & (1 << col):
                pivot = r
                break
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        for r in range(m):
            if r != rank and aug[r] & (1 << col):
                aug[r] ^= aug[rank]
        rank += 1

    # Inconsistent if any row has 0 coefficients but nonzero RHS
    for r in range(rank, m):
        if aug[r] & (1 << n):
            return True
    return False


def kxor_hard_axioms(n, k=3, clause_ratio=2.0, seed=42):
    """Hard k-XOR with global GF(2) contradiction.

    Generates random k-XOR clauses and finds a RHS vector that makes
    the system unsatisfiable over GF(2). The contradiction is spread
    across all clauses (no single-clause flip trick).
    """
    rng = random.Random(seed)
    num_clauses = max(k + 1, int(clause_ratio * n))

    # Generate random clause structure
    clauses_vars = []
    for _ in range(num_clauses):
        clauses_vars.append(sorted(rng.sample(range(n), k)))

    # Find an unsatisfiable RHS via random search
    # With m > n, most RHS vectors are unsat
    b_vec = None
    for _ in range(200):
        candidate = [rng.randint(0, 1) for _ in range(num_clauses)]
        if _gf2_is_unsat(clauses_vars, candidate, n):
            b_vec = candidate
            break

    if b_vec is None:
        # Fallback: planted + flip ALL clauses (global contradiction)
        a = [rng.randint(0, 1) for _ in range(n)]
        b_vec = []
        for vars_chosen in clauses_vars:
            parity = 0
            for v in vars_chosen:
                parity ^= a[v]
            b_vec.append(parity ^ 1)  # flip all

    # Encode as polynomial axioms
    axioms = []
    for vars_chosen, bi in zip(clauses_vars, b_vec):
        terms = [(-float(bi), frozenset())]
        for j in range(1, k + 1):
            coeff = (-2.0) ** (j - 1)
            for subset in combinations(vars_chosen, j):
                terms.append((coeff, frozenset(subset)))
        axioms.append(terms)

    return axioms, n


# =====================================================================
# SUBSET-SUM BUILDER
# =====================================================================

def subset_sum_axioms(n_items, max_weight=4, seed=42):
    """Subset-Sum insatisfacible: pesos pares, target impar."""
    rng = random.Random(seed)
    weights = [2 * rng.randint(1, max_weight // 2) for _ in range(n_items)]
    target = 1
    capacity = sum(weights)

    idx = 0
    var_x = {}
    for i in range(n_items):
        var_x[i] = idx
        idx += 1

    var_c = {}
    for i in range(n_items + 1):
        for s in range(capacity + 1):
            var_c[(i, s)] = idx
            idx += 1
    num_vars = idx

    axioms = []
    axioms.append([(-1.0, frozenset()), (1.0, frozenset([var_c[(0, 0)]]))])
    for s in range(1, capacity + 1):
        axioms.append([(1.0, frozenset([var_c[(0, s)]]))])

    for i in range(1, n_items + 1):
        w_i = weights[i - 1]
        xi = var_x[i - 1]
        for s in range(capacity + 1):
            c_is = var_c[(i, s)]
            c_prev_s = var_c[(i - 1, s)]
            terms = [
                (1.0, frozenset([c_is])),
                (-1.0, frozenset([c_prev_s])),
                (1.0, frozenset([xi, c_prev_s])),
            ]
            if s - w_i >= 0:
                c_prev_sw = var_c[(i - 1, s - w_i)]
                terms.append((-1.0, frozenset([xi, c_prev_sw])))
            axioms.append(terms)

    axioms.append([
        (-1.0, frozenset()),
        (1.0, frozenset([var_c[(n_items, target)]])),
    ])
    return axioms, num_vars


# =====================================================================
# FAMILY DEFINITIONS (with adaptive max_degree)
# =====================================================================

FAMILIES = [
    {
        'name': 'PHP',
        'description': 'Pigeonhole Principle (canonical SC(0))',
        'expected_sc': 0,
        'builder': lambda n: php_axioms(n)[:2],
        'n_values': [2, 3, 4],
        'max_degree': 10,   # PHP(4) needs d=8
    },
    {
        'name': 'PHP-E',
        'description': 'PHP-Entrelazado with y-variables (canonical SC(1))',
        'expected_sc': 1,
        'builder': lambda n: phpe_axioms(n)[:2],
        'n_values': [2, 3, 4],
        'max_degree': 8,
    },
    {
        'name': 'PHP-C',
        'description': 'PHP-Circular with s-variables (canonical SC(2))',
        'expected_sc': 2,
        'builder': lambda n: phpc_axioms(n)[:2],
        'n_values': [2, 3],
        'max_degree': 8,    # PHP-C(3) caps at d=6 anyway (24 vars)
    },
    {
        'name': 'Tseitin-cycle',
        'description': 'Tseitin parity on cycle graphs',
        'expected_sc': '0-1',
        'builder': lambda n: tseitin_axioms(*circulant_graph(n, [1])),
        'n_values': [6, 8, 10, 12],
        'max_degree': 8,
    },
    {
        'name': 'Tseitin-expander',
        'description': 'Tseitin parity on 4-regular circulant C_n(1,3)',
        'expected_sc': '2-3?',
        'builder': lambda n: tseitin_axioms(*circulant_graph(n, [1, 3])),
        'n_values': [8, 10, 12],
        'max_degree': 8,
    },
    {
        'name': '3-XOR',
        'description': 'Random 3-XOR with GF(2) global contradiction',
        'expected_sc': 0,
        'builder': lambda n: kxor_hard_axioms(n),
        'n_values': [8, 10, 12, 15],
        'max_degree': 8,
    },
    {
        'name': 'Subset-Sum',
        'description': 'Subset-Sum DP (even weights, odd target)',
        'expected_sc': 0,
        'builder': lambda n: subset_sum_axioms(n),
        'n_values': [2, 3],
        'max_degree': 6,    # many vars from DP table
    },
    {
        'name': 'Factoring',
        'description': 'Integer factoring (prime > 2^n)',
        'expected_sc': '1-2?',
        'builder': lambda n: factoring_axioms(n)[:2],
        'n_values': [4, 6],
        'max_degree': 6,
    },
    {
        'name': 'Goldreich-PRG',
        'description': 'Goldreich PRG inversion (P5 predicate)',
        'expected_sc': '2-3?',
        'builder': lambda n: goldreich_prg_axioms(n)[:2],
        'n_values': [8, 12],
        'max_degree': 5,
    },
    {
        'name': 'Binary-LWE',
        'description': 'Binary LWE with Hamming weight bound',
        'expected_sc': '1-2?',
        'builder': lambda n: binary_lwe_axioms(n)[:2],
        'n_values': [4, 6],
        'max_degree': 4,
    },
]


# =====================================================================
# MAIN MAPPING
# =====================================================================

def run_landscape():
    t_start = time.time()

    print("=" * 76)
    print("  SELECTOR COMPLEXITY LANDSCAPE MAP v3")
    print("  10 families — adaptive degree — family-level classification")
    print("=" * 76)
    print()

    family_summaries = []
    comparison_systems = []

    for fam in FAMILIES:
        print()
        print("#" * 76)
        print("##  {} — {}".format(fam['name'], fam['description']))
        print("##  Expected: SC({})   max_degree={}".format(
            fam['expected_sc'], fam['max_degree']))
        print("##  Sizes: n = {}".format(fam['n_values']))
        print("#" * 76)
        print()

        t_fam = time.time()
        md = fam['max_degree']

        # ── 1. Family-level SC classification (the reliable one) ──
        print("  [Family classifier]")
        fam_result = estimate_level_family(
            fam['builder'], fam['n_values'],
            max_degree=md, verbose=True)
        family_sc = fam_result['level']
        family_conf = fam_result['confidence']
        print()

        # ── 2. Per-instance hardness scores ──
        print("  [Per-instance hardness]")
        scores = []
        for n in fam['n_values']:
            result = fam['builder'](n)
            ax, nv = result[0], result[1]
            h = quantify_hardness(ax, nv, max_degree=md)
            scores.append({
                'n': n,
                'hardness_score': h['hardness_score'],
                'sc_level': h['sc_level'],
                'ips_degree': h['ips_degree'],
                'expansion': h['expansion'],
                'num_vars': nv,
                'num_axioms': len(ax),
            })
            ips_str = str(h['ips_degree']) if h['ips_degree'] else "N/A"
            print("    n={}: hardness={:.1f}, SC({}), IPS_d={}, vars={}".format(
                n, h['hardness_score'], h['sc_level'], ips_str, nv))
        print()

        # ── 3. Strategy for smallest and largest ──
        n_small = fam['n_values'][0]
        ax_s, nv_s = fam['builder'](n_small)[0], fam['builder'](n_small)[1]
        strat_small = recommend_strategy(ax_s, nv_s, max_degree=min(md, 5))

        n_big = fam['n_values'][-1]
        ax_b, nv_b = fam['builder'](n_big)[0], fam['builder'](n_big)[1]
        strat_big = recommend_strategy(ax_b, nv_b, max_degree=min(md, 5))

        print("  Strategy n={}: {} (SC {})".format(
            n_small, strat_small['strategy'], strat_small['sc_level']))
        print("  Strategy n={}: {} (SC {})".format(
            n_big, strat_big['strategy'], strat_big['sc_level']))

        elapsed_fam = time.time() - t_fam

        summary = {
            'name': fam['name'],
            'expected_sc': fam['expected_sc'],
            'family_sc': family_sc,
            'family_conf': family_conf,
            'n_range': fam['n_values'],
            'scaling': fam_result.get('scaling', {}),
            'min_hardness': min(s['hardness_score'] for s in scores),
            'max_hardness': max(s['hardness_score'] for s in scores),
            'strategy_small': strat_small['strategy'],
            'strategy_large': strat_big['strategy'],
            'time': round(elapsed_fam, 1),
            'scores': scores,
        }
        family_summaries.append(summary)

        # For combined ranking
        comparison_systems.append({
            'name': '{}({})'.format(fam['name'], n_big),
            'axioms': ax_b,
            'num_vars': nv_b,
        })

        print()
        print("  FAMILY RESULT: SC({}) [confidence: {}]  [{:.1f}s]".format(
            family_sc, family_conf, elapsed_fam))
        print()

    # =====================================================================
    # COMBINED RANKING
    # =====================================================================

    print()
    print("#" * 76)
    print("##  COMBINED HARDNESS RANKING (largest instance per family)")
    print("#" * 76)
    print()

    # Use per-family max_degree for the comparison too
    results_for_ranking = []
    for fam, sys in zip(FAMILIES, comparison_systems):
        h = quantify_hardness(sys['axioms'], sys['num_vars'],
                              max_degree=fam['max_degree'])
        results_for_ranking.append({
            'name': sys['name'],
            'hardness': h['hardness_score'],
            'sc_level': h['sc_level'],
            'ips_degree': h['ips_degree'],
            'expansion': h['expansion'],
        })

    results_for_ranking.sort(key=lambda x: x['hardness'], reverse=True)

    print("  {:>4}  {:<22} {:>8} {:>8} {:>8} {:>8}".format(
        "Rank", "System", "Score", "SC", "IPS_d", "Expan"))
    print("  " + "-" * 62)
    for i, r in enumerate(results_for_ranking, 1):
        ips_str = str(r['ips_degree']) if r['ips_degree'] is not None else "N/A"
        sc_str = "SC({})".format(r['sc_level'])
        print("  {:>4}  {:<22} {:>8.1f} {:>8} {:>8} {:>8.3f}".format(
            i, r['name'], r['hardness'], sc_str, ips_str, r['expansion']))

    # =====================================================================
    # LANDSCAPE SUMMARY
    # =====================================================================

    print()
    print()
    print("#" * 76)
    print("##  LANDSCAPE SUMMARY (family-level classification)")
    print("#" * 76)
    print()

    print("  {:<20} {:>6} {:>8} {:>10} {:>10} {:>14} {:>8}".format(
        "Family", "SC_exp", "SC_obs", "Confid", "Hardness", "Growth(deg)", "Time"))
    print("  " + "-" * 80)

    for s in family_summaries:
        h_range = "{:.0f}-{:.0f}".format(s['min_hardness'], s['max_hardness'])
        deg_growth = s['scaling'].get('degree_growth', '?')
        print("  {:<20} {:>6} {:>8} {:>10} {:>10} {:>14} {:>7.0f}s".format(
            s['name'],
            str(s['expected_sc']),
            "SC({})".format(s['family_sc']),
            s['family_conf'],
            h_range,
            str(deg_growth)[:14],
            s['time'],
        ))

    # =====================================================================
    # SC HIERARCHY PLACEMENT (using FAMILY classifier)
    # =====================================================================

    print()
    print()
    print("#" * 76)
    print("##  SC HIERARCHY PLACEMENT (family-level)")
    print("#" * 76)
    print()

    sc_groups = {}
    for s in family_summaries:
        level = s['family_sc']
        key = str(level)
        if key not in sc_groups:
            sc_groups[key] = []
        sc_groups[key].append(s['name'])

    for level in ['0', '1', '2', '3?']:
        members = sc_groups.get(level, [])
        if members:
            print("  SC({:>2}):  {}".format(level, ", ".join(members)))
        else:
            print("  SC({:>2}):  (none observed)".format(level))

    # =====================================================================
    # GROWTH PATTERNS
    # =====================================================================

    print()
    print()
    print("#" * 76)
    print("##  GROWTH PATTERNS")
    print("#" * 76)
    print()

    for s in family_summaries:
        scores_str = ", ".join(
            "n={}:{:.0f}".format(sc['n'], sc['hardness_score'])
            for sc in s['scores'])
        deg_growth = s['scaling'].get('degree_growth', '?')
        size_growth = s['scaling'].get('size_growth', '?')
        print("  {:<20} hardness: [{}]".format(s['name'], scores_str))
        print("  {:<20} degree_growth={}, size_growth={}".format(
            '', deg_growth, size_growth))
        print()

    total_time = time.time() - t_start
    print("=" * 76)
    print("  Total time: {:.1f}s".format(total_time))
    print("=" * 76)


if __name__ == "__main__":
    run_landscape()
