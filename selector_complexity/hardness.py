"""
Hardness Quantifier
===================

Quantifies the proof-theoretic hardness of polynomial systems
using a composite score (0-100) based on:

    - Selector Complexity level (40%)
    - IPS certificate degree (30%)
    - Structural properties (15%)
    - Constraint graph expansion (15%)

Functions:
    quantify_hardness  -- single-system hardness score
    compare_hardness   -- rank multiple systems
    hardness_report    -- family-level hardness analysis

Author: Carmen Esteban
License: MIT
"""

from selector_complexity.classifier import (
    _analyze_structure,
    _fit_growth_rate,
    estimate_level,
)


# =====================================================================
# EXPANSION METRIC
# =====================================================================

def _compute_expansion(axioms, num_vars):
    """Compute expansion of the constraint graph.

    Expansion measures how well-connected the constraint graph is.
    Higher expansion means the system is harder to decompose.

    Uses average normalized degree as a proxy for vertex expansion.

    Parameters
    ----------
    axioms : list
        Polynomial axioms.
    num_vars : int
        Number of variables.

    Returns
    -------
    float
        Expansion value in [0, 1].
    """
    if num_vars <= 1:
        return 0.0

    # Build adjacency from axioms
    neighbors = {v: set() for v in range(num_vars)}
    for ax in axioms:
        ax_vars = set()
        for _, m in ax:
            ax_vars.update(m)
        ax_vars = list(ax_vars)
        for i, v1 in enumerate(ax_vars):
            for v2 in ax_vars[i + 1:]:
                if v1 in neighbors:
                    neighbors[v1].add(v2)
                if v2 in neighbors:
                    neighbors[v2].add(v1)

    # Average degree / (num_vars - 1) as normalized expansion
    expansions = []
    for v in range(num_vars):
        if v in neighbors and neighbors[v]:
            expansions.append(len(neighbors[v]))

    if not expansions:
        return 0.0

    avg_degree = sum(expansions) / len(expansions)
    normalized = avg_degree / max(num_vars - 1, 1)
    return min(normalized, 1.0)


# =====================================================================
# SINGLE-SYSTEM HARDNESS
# =====================================================================

def quantify_hardness(axioms, num_vars, max_degree=6):
    """Compute a hardness profile for a polynomial system.

    Score components (0-100):
        - SC level:   40% (level 0 -> 0, 1 -> 13, 2 -> 27, 3? -> 40)
        - IPS degree: 30% (scaled by degree gap relative to axiom degree)
        - Structure:  15% (connectivity + variable density)
        - Expansion:  15% (constraint graph expansion)

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        The polynomial axioms.
    num_vars : int
        Number of variables.
    max_degree : int, optional
        Maximum certificate degree to search (default 6).

    Returns
    -------
    dict
        'hardness_score' : float
            Composite score 0-100.
        'sc_level' : int or str
            Estimated SC level.
        'ips_degree' : int or None
            Minimum certificate degree found (None if not found).
        'expansion' : float
            Constraint graph expansion (0-1).
        'score_breakdown' : dict
            Individual component scores.
    """
    # Run classification (quiet)
    result = estimate_level(axioms, num_vars, max_degree=max_degree, verbose=False)
    level = result['level']
    evidence = result['evidence']

    structure = evidence['structure']
    certs = evidence['certificates']
    feasible = [c for c in certs if c['feasible']]

    # Component 1: SC level (40%)
    level_scores = {0: 0, 1: 13, 2: 27, '3?': 40}
    sc_score = level_scores.get(level, 40)

    # Component 2: IPS degree (30%)
    max_axiom_deg = structure.get('max_axiom_degree', 0)
    if feasible:
        ips_degree = feasible[0]['degree']
        degree_gap = ips_degree - max_axiom_deg
        # Scale: gap 0 -> 0pts, gap 10+ -> 30pts
        ips_score = min(degree_gap * 3.0, 30.0)
    else:
        ips_degree = None
        ips_score = 30.0  # no certificate -> max hardness for this component

    # Component 3: Structure (15%)
    connectivity = structure.get('connectivity', 0)
    var_density = structure.get('var_density', 0)
    struct_score = connectivity * 10.0 + var_density * 5.0
    struct_score = min(struct_score, 15.0)

    # Component 4: Expansion (15%)
    expansion = _compute_expansion(axioms, num_vars)
    expansion_score = expansion * 15.0

    # Total
    hardness = sc_score + ips_score + struct_score + expansion_score
    hardness = round(min(hardness, 100.0), 1)

    return {
        'hardness_score': hardness,
        'sc_level': level,
        'ips_degree': ips_degree,
        'expansion': round(expansion, 4),
        'score_breakdown': {
            'sc_level_score': round(sc_score, 1),
            'ips_degree_score': round(ips_score, 1),
            'structure_score': round(struct_score, 1),
            'expansion_score': round(expansion_score, 1),
        },
    }


# =====================================================================
# MULTI-SYSTEM COMPARISON
# =====================================================================

def compare_hardness(systems, max_degree=6):
    """Rank multiple polynomial systems by hardness.

    Parameters
    ----------
    systems : list of dict
        Each dict: {'name': str, 'axioms': list, 'num_vars': int}
    max_degree : int, optional
        Maximum certificate degree to search (default 6).

    Returns
    -------
    str
        ASCII table ranking systems by hardness score (descending).
    """
    results = []
    for sys in systems:
        h = quantify_hardness(sys['axioms'], sys['num_vars'], max_degree)
        results.append({
            'name': sys['name'],
            'hardness': h['hardness_score'],
            'sc_level': h['sc_level'],
            'ips_degree': h['ips_degree'],
            'expansion': h['expansion'],
        })

    # Sort by hardness descending
    results.sort(key=lambda x: x['hardness'], reverse=True)

    # Build ASCII table
    lines = []
    lines.append("=" * 68)
    lines.append("  HARDNESS RANKING")
    lines.append("=" * 68)
    lines.append("")
    lines.append("  {:>4}  {:<20} {:>8} {:>8} {:>8} {:>8}".format(
        "Rank", "System", "Score", "SC", "IPS_d", "Expan"))
    lines.append("  " + "-" * 62)

    for i, r in enumerate(results, 1):
        ips_str = str(r['ips_degree']) if r['ips_degree'] is not None else "N/A"
        sc_str = "SC({})".format(r['sc_level'])
        lines.append("  {:>4}  {:<20} {:>8.1f} {:>8} {:>8} {:>8.3f}".format(
            i, r['name'], r['hardness'], sc_str, ips_str, r['expansion']))

    lines.append("")
    lines.append("=" * 68)
    return "\n".join(lines)


# =====================================================================
# FAMILY-LEVEL HARDNESS REPORT
# =====================================================================

def hardness_report(builder, n_values, max_degree=6):
    """Analyze hardness scaling across a family of instances.

    Parameters
    ----------
    builder : callable
        Function n -> (axioms, num_vars) or n -> (axioms, num_vars, ...).
    n_values : list of int
        Parameter values to test.
    max_degree : int, optional
        Maximum certificate degree per instance (default 6).

    Returns
    -------
    dict
        'scores' : list of dict
            Per-instance hardness results.
        'growth_pattern' : str
            'polynomial', 'exponential', 'factorial', or other.
        'scaling_table' : str
            ASCII table of scores by n.
    """
    scores = []
    for n in n_values:
        result = builder(n)
        axioms = result[0]
        num_vars = result[1]

        h = quantify_hardness(axioms, num_vars, max_degree)
        scores.append({
            'n': n,
            'hardness_score': h['hardness_score'],
            'sc_level': h['sc_level'],
            'ips_degree': h['ips_degree'],
            'expansion': h['expansion'],
            'num_vars': num_vars,
            'num_axioms': len(axioms),
        })

    # Fit growth rate on hardness scores
    if len(n_values) >= 2:
        hardness_values = [s['hardness_score'] for s in scores]
        # Avoid log(0)
        safe_values = [max(v, 0.01) for v in hardness_values]
        growth_pattern, _ = _fit_growth_rate(n_values, safe_values)
    else:
        growth_pattern = 'insufficient_data'

    # Build ASCII table
    lines = []
    lines.append("=" * 72)
    lines.append("  HARDNESS SCALING REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  {:>4}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "n", "vars", "axioms", "Score", "SC", "IPS_d", "Expan"))
    lines.append("  " + "-" * 66)

    for s in scores:
        ips_str = str(s['ips_degree']) if s['ips_degree'] is not None else "N/A"
        sc_str = "SC({})".format(s['sc_level'])
        lines.append("  {:>4}  {:>6}  {:>8}  {:>8.1f}  {:>8}  {:>8}  {:>8.3f}".format(
            s['n'], s['num_vars'], s['num_axioms'],
            s['hardness_score'], sc_str, ips_str, s['expansion']))

    lines.append("")
    lines.append("  Growth pattern: {}".format(growth_pattern))
    lines.append("=" * 72)

    scaling_table = "\n".join(lines)

    return {
        'scores': scores,
        'growth_pattern': growth_pattern,
        'scaling_table': scaling_table,
    }
