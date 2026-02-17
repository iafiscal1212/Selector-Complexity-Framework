"""
Pattern Detection and Shortcuts
================================

Detects known structural patterns in polynomial axiom systems to
short-circuit expensive IPS certificate search.

Detectors:
    detect_xor_structure        -- k-XOR encoding pattern
    detect_subset_sum_structure -- DP recurrence pattern
    detect_graph_topology       -- constraint graph topology
    detect_patterns             -- combined dispatcher

Author: Carmen Esteban
License: MIT
"""

from itertools import combinations
from math import comb


# =====================================================================
# XOR STRUCTURE DETECTOR
# =====================================================================

def detect_xor_structure(axioms, num_vars):
    """Detect if axioms follow the k-XOR encoding pattern.

    The k-XOR encoding uses (-2)^{j-1} * e_j(vars) coefficients.
    If >= 80% of axioms match this pattern, classify as XOR-type.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    dict
        'is_xor' : bool
        'k' : int or None (detected clause width)
        'match_ratio' : float (fraction of axioms matching)
        'confidence' : str
    """
    if not axioms:
        return {'is_xor': False, 'k': None, 'match_ratio': 0.0, 'confidence': 'low'}

    matches = 0
    detected_k = None

    for ax in axioms:
        # Extract non-constant terms and their degrees
        terms_by_degree = {}
        constant_term = 0.0
        for coef, mono in ax:
            deg = len(mono)
            if deg == 0:
                constant_term += coef
            else:
                if deg not in terms_by_degree:
                    terms_by_degree[deg] = []
                terms_by_degree[deg].append((coef, mono))

        if not terms_by_degree:
            continue

        max_deg = max(terms_by_degree.keys())

        # Collect variables used in highest-degree term
        all_vars = set()
        for coef, mono in ax:
            all_vars.update(mono)

        k = len(all_vars)
        if k < 2:
            continue

        # Check XOR pattern: for degree j, expect C(k,j) terms
        # each with coefficient (-2)^{j-1}
        is_match = True
        for j in range(1, k + 1):
            expected_count = comb(k, j)
            expected_coef_abs = 2.0 ** (j - 1)

            actual = terms_by_degree.get(j, [])
            if len(actual) != expected_count:
                is_match = False
                break

            for coef, mono in actual:
                if abs(abs(coef) - expected_coef_abs) > 1e-10:
                    is_match = False
                    break
            if not is_match:
                break

        if is_match:
            matches += 1
            if detected_k is None:
                detected_k = k
            elif k != detected_k:
                detected_k = k  # use the latest

    match_ratio = matches / len(axioms)
    is_xor = match_ratio >= 0.80

    if is_xor:
        confidence = 'high' if match_ratio >= 0.95 else 'medium'
    else:
        confidence = 'low'

    return {
        'is_xor': is_xor,
        'k': detected_k,
        'match_ratio': round(match_ratio, 3),
        'confidence': confidence,
    }


# =====================================================================
# SUBSET-SUM STRUCTURE DETECTOR
# =====================================================================

def detect_subset_sum_structure(axioms, num_vars):
    """Detect axioms with DP recurrence pattern (Subset-Sum-like).

    DP recurrences typically have 3-4 terms, degrees 1-2, with
    alternating signs. If > 50% match + base cases detected,
    classify as Subset-Sum type.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    dict
        'is_subset_sum' : bool
        'match_ratio' : float
        'has_base_cases' : bool
        'estimated_sc' : int or None
        'confidence' : str
    """
    if not axioms:
        return {
            'is_subset_sum': False, 'match_ratio': 0.0,
            'has_base_cases': False, 'estimated_sc': None, 'confidence': 'low',
        }

    dp_matches = 0
    base_cases = 0

    for ax in axioms:
        num_terms = len(ax)
        max_deg = max((len(m) for _, m in ax), default=0)

        # Base case: 1-2 terms, one constant, degrees 0-1
        if num_terms <= 2 and max_deg <= 1:
            has_const = any(len(m) == 0 for _, m in ax)
            if has_const:
                base_cases += 1
                continue

        # DP recurrence pattern: 3-4 terms, max degree 2, alternating signs
        if 3 <= num_terms <= 4 and max_deg <= 2:
            signs = [1 if c > 0 else -1 for c, _ in ax]
            has_positive = any(s > 0 for s in signs)
            has_negative = any(s < 0 for s in signs)
            # All coefficients are +1 or -1
            all_unit = all(abs(abs(c) - 1.0) < 1e-10 for c, _ in ax)

            if has_positive and has_negative and all_unit:
                dp_matches += 1

    total = len(axioms)
    match_ratio = (dp_matches + base_cases) / total if total > 0 else 0.0
    has_base = base_cases >= 2

    is_subset_sum = match_ratio > 0.50 and has_base

    if is_subset_sum:
        estimated_sc = 1 if num_vars < 50 else 2
        confidence = 'medium' if match_ratio > 0.70 else 'low'
    else:
        estimated_sc = None
        confidence = 'low'

    return {
        'is_subset_sum': is_subset_sum,
        'match_ratio': round(match_ratio, 3),
        'has_base_cases': has_base,
        'estimated_sc': estimated_sc,
        'confidence': confidence,
    }


# =====================================================================
# GRAPH TOPOLOGY DETECTOR
# =====================================================================

def detect_graph_topology(axioms, num_vars):
    """Classify the constraint graph topology.

    Builds a variable adjacency graph from axioms and classifies it:
    - 'cycle': regular degree 2, single component -> SC(0)
    - 'tree': connected, |E| = |V| - 1 -> SC(0)
    - 'expander': high min-degree, few components -> SC(2-3?)
    - 'complete': density > 0.8 -> SC(2)
    - 'sparse': low connectivity
    - 'unknown': none of the above

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    dict
        'topology' : str
        'num_components' : int
        'avg_degree' : float
        'min_degree' : int
        'max_degree' : int
        'density' : float
        'estimated_sc' : int or str or None
        'confidence' : str
    """
    if not axioms or num_vars == 0:
        return {
            'topology': 'empty', 'num_components': 0,
            'avg_degree': 0.0, 'min_degree': 0, 'max_degree': 0,
            'density': 0.0, 'estimated_sc': None, 'confidence': 'low',
        }

    # Build adjacency from axioms
    adj = {}
    for v in range(num_vars):
        adj[v] = set()

    for ax in axioms:
        ax_vars = set()
        for _, m in ax:
            ax_vars.update(m)
        ax_vars_list = list(ax_vars)
        for i, v1 in enumerate(ax_vars_list):
            for v2 in ax_vars_list[i + 1:]:
                if v1 < num_vars and v2 < num_vars:
                    adj.setdefault(v1, set()).add(v2)
                    adj.setdefault(v2, set()).add(v1)

    # Degree statistics
    degrees = [len(adj.get(v, set())) for v in range(num_vars)]
    active_degrees = [d for d in degrees if d > 0]

    if not active_degrees:
        return {
            'topology': 'isolated', 'num_components': num_vars,
            'avg_degree': 0.0, 'min_degree': 0, 'max_degree': 0,
            'density': 0.0, 'estimated_sc': 0, 'confidence': 'low',
        }

    avg_deg = sum(active_degrees) / len(active_degrees)
    min_deg = min(active_degrees)
    max_deg = max(active_degrees)

    # Count edges
    num_edges = sum(degrees) // 2
    active_vars = len(active_degrees)
    max_edges = active_vars * (active_vars - 1) // 2
    density = num_edges / max(max_edges, 1)

    # Connected components via BFS
    visited = set()
    components = []
    for start in range(num_vars):
        if start in visited or not adj.get(start):
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        components.append(component)

    num_components = len(components)

    # Classify topology
    topology = 'unknown'
    estimated_sc = None
    confidence = 'low'

    # Cycle: single component, all vertices degree 2
    if (num_components == 1 and active_vars >= 3
            and min_deg == 2 and max_deg == 2):
        topology = 'cycle'
        estimated_sc = 0
        confidence = 'high'

    # Tree: single component, |E| = |V| - 1
    elif num_components == 1 and num_edges == active_vars - 1:
        topology = 'tree'
        estimated_sc = 0
        confidence = 'medium'

    # Complete: very high density
    elif density > 0.80:
        topology = 'complete'
        estimated_sc = 2
        confidence = 'medium'

    # Expander: high min-degree, few components, moderate density
    elif (min_deg >= 3 and num_components <= 2
            and density > 0.1 and active_vars >= 6):
        topology = 'expander'
        estimated_sc = '2-3?'
        confidence = 'medium'

    # Sparse
    elif density < 0.15:
        topology = 'sparse'
        estimated_sc = 0
        confidence = 'low'

    return {
        'topology': topology,
        'num_components': num_components,
        'avg_degree': round(avg_deg, 2),
        'min_degree': min_deg,
        'max_degree': max_deg,
        'density': round(density, 4),
        'estimated_sc': estimated_sc,
        'confidence': confidence,
    }


# =====================================================================
# COMBINED PATTERN DISPATCHER
# =====================================================================

def detect_patterns(axioms, num_vars):
    """Run all pattern detectors and decide if a shortcut is available.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    dict
        'shortcut_available' : bool
        'shortcut_level' : int or str or None
        'shortcut_confidence' : str or None
        'shortcut_source' : str or None (which detector triggered)
        'xor' : dict (detect_xor_structure result)
        'subset_sum' : dict (detect_subset_sum_structure result)
        'topology' : dict (detect_graph_topology result)
    """
    xor_result = detect_xor_structure(axioms, num_vars)
    ss_result = detect_subset_sum_structure(axioms, num_vars)
    topo_result = detect_graph_topology(axioms, num_vars)

    shortcut = False
    level = None
    confidence = None
    source = None

    # Priority 1: XOR pattern with high confidence
    if xor_result['is_xor'] and xor_result['confidence'] in ('high', 'medium'):
        # XOR on cycle graphs -> SC(0), on expanders -> higher
        if topo_result['topology'] == 'cycle':
            shortcut = True
            level = 0
            confidence = 'high'
            source = 'xor+cycle'
        elif topo_result['topology'] == 'expander':
            shortcut = True
            level = '2-3?'
            confidence = 'medium'
            source = 'xor+expander'
        else:
            # Generic XOR: depends on graph structure
            shortcut = True
            level = 0
            confidence = 'medium'
            source = 'xor'

    # Priority 2: Graph topology with high confidence
    elif topo_result['confidence'] == 'high' and topo_result['estimated_sc'] is not None:
        shortcut = True
        level = topo_result['estimated_sc']
        confidence = topo_result['confidence']
        source = 'topology:' + topo_result['topology']

    # Priority 3: Subset-Sum pattern
    elif ss_result['is_subset_sum'] and ss_result['confidence'] in ('medium',):
        shortcut = True
        level = ss_result['estimated_sc']
        confidence = ss_result['confidence']
        source = 'subset_sum'

    return {
        'shortcut_available': shortcut,
        'shortcut_level': level,
        'shortcut_confidence': confidence,
        'shortcut_source': source,
        'xor': xor_result,
        'subset_sum': ss_result,
        'topology': topo_result,
    }
