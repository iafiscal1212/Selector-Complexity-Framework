"""
Proof Strategy Advisor
======================

Recommends the best proof strategy for a polynomial system
based on its Selector Complexity level.

Strategies:
    direct_ips              -> SC(0): search for a direct certificate
    selector_decomposition  -> SC(1): build selectors, decompose
    exhaustive_with_pruning -> SC(2): enumeration with symmetry-breaking
    algebraic_global        -> SC(3?): global methods, avoid decomposition

Author: Carmen Esteban
License: MIT
"""

import time

from selector_complexity.classifier import _analyze_structure, _search_certificates


def recommend_strategy(axioms, num_vars, max_degree=4):
    """Recommend a proof strategy based on structural and IPS analysis.

    Runs a quick structural analysis followed by a low-degree IPS
    certificate search, then selects the most appropriate strategy.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        The polynomial axioms.
    num_vars : int
        Number of variables.
    max_degree : int, optional
        Maximum degree for quick IPS search (default 4).

    Returns
    -------
    dict
        'strategy' : str
            One of: 'direct_ips', 'selector_decomposition',
            'exhaustive_with_pruning', 'algebraic_global'.
        'sc_level' : int or str
            Estimated SC level (0, 1, 2, or '3?').
        'parameters' : dict
            Suggested parameters for the chosen strategy.
        'reasoning' : str
            Explanation of why this strategy was chosen.
    """
    t0 = time.time()

    # Phase 1: structural analysis
    structure = _analyze_structure(axioms, num_vars)

    # Phase 2: quick IPS search at low degree
    certificates = _search_certificates(
        axioms, num_vars, max_degree, verbose=False)

    elapsed = time.time() - t0

    feasible = [c for c in certificates if c['feasible']]
    max_axiom_deg = structure['max_axiom_degree']
    connectivity = structure['connectivity']

    # ── Certificate found ──
    if feasible:
        best = feasible[0]
        degree_gap = best['degree'] - max_axiom_deg

        if degree_gap <= 3:
            # SC(0): direct certificate works
            return {
                'strategy': 'direct_ips',
                'sc_level': 0,
                'parameters': {
                    'recommended_degree': best['degree'],
                    'expected_size': best['size'],
                    'search_time': round(elapsed, 4),
                },
                'reasoning': (
                    "Certificate found at degree {} (gap {}). "
                    "Direct IPS search is sufficient — no selectors needed. "
                    "Recommend searching at degree {}.".format(
                        best['degree'], degree_gap, best['degree'])
                ),
            }

        if degree_gap <= 6:
            # SC(1): selectors would help
            return {
                'strategy': 'selector_decomposition',
                'sc_level': 1,
                'parameters': {
                    'baseline_degree': best['degree'],
                    'recommended_selector_degree': min(degree_gap, 4),
                    'max_branches': min(num_vars, 10),
                    'search_time': round(elapsed, 4),
                },
                'reasoning': (
                    "Certificate found at degree {} but with gap {}. "
                    "Selector decomposition can reduce the effective degree. "
                    "Try building selectors of degree <= {} with "
                    "up to {} branches.".format(
                        best['degree'], degree_gap,
                        min(degree_gap, 4), min(num_vars, 10))
                ),
            }

        # SC(2): certificate exists but expensive
        return {
            'strategy': 'exhaustive_with_pruning',
            'sc_level': 2,
            'parameters': {
                'baseline_degree': best['degree'],
                'symmetry_breaking': True,
                'pruning_threshold': 3,
                'max_branches': num_vars,
                'search_time': round(elapsed, 4),
            },
            'reasoning': (
                "Certificate found at degree {} (gap {}). "
                "Large gap means selectors exist but are expensive. "
                "Use exhaustive search with symmetry-breaking and "
                "pruning (stop after {} non-improving rounds).".format(
                    best['degree'], degree_gap, 3)
            ),
        }

    # ── No certificate found ──
    # Check residual trend
    residuals = [c['residual'] for c in certificates if c['residual'] < 1.0]
    improving = False
    if len(residuals) >= 2:
        improving = residuals[-1] < residuals[0] * 0.5

    if improving:
        # Residuals improving -> might find at higher degree -> SC(2)
        return {
            'strategy': 'exhaustive_with_pruning',
            'sc_level': 2,
            'parameters': {
                'recommended_max_degree': max_degree + 4,
                'symmetry_breaking': True,
                'pruning_threshold': 3,
                'residual_trend': [round(r, 6) for r in residuals],
                'search_time': round(elapsed, 4),
            },
            'reasoning': (
                "No certificate found up to degree {}, but residuals are "
                "improving ({:.2e} -> {:.2e}). Try higher degrees with "
                "exhaustive search and symmetry-breaking.".format(
                    max_degree, residuals[0], residuals[-1])
            ),
        }

    if connectivity > 0.3:
        # High connectivity -> expander-like -> SC(3?)
        return {
            'strategy': 'algebraic_global',
            'sc_level': '3?',
            'parameters': {
                'connectivity': round(connectivity, 4),
                'avoid_decomposition': True,
                'try_groebner': num_vars <= 20,
                'try_sum_of_squares': True,
                'search_time': round(elapsed, 4),
            },
            'reasoning': (
                "No certificate found up to degree {} and high connectivity "
                "({:.2f}) suggests expander-like structure. "
                "Decomposition strategies are unlikely to help. "
                "Try global algebraic methods (Groebner bases, SOS).".format(
                    max_degree, connectivity)
            ),
        }

    # Default: algebraic global
    return {
        'strategy': 'algebraic_global',
        'sc_level': '3?',
        'parameters': {
            'connectivity': round(connectivity, 4),
            'try_higher_degree': True,
            'recommended_max_degree': max_degree + 6,
            'search_time': round(elapsed, 4),
        },
        'reasoning': (
            "No certificate found up to degree {} and no clear improvement "
            "trend. Structure does not suggest a particular decomposition. "
            "Try global methods or increase search degree.".format(max_degree)
        ),
    }
