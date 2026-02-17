"""
Automatic SC-Level Classifier
==============================

Estimates the Selector Complexity level of an arbitrary polynomial system
by running a battery of structural and computational tests.

Two modes:

  1. Single instance:   estimate_level(axioms, num_vars)
  2. Family scaling:    estimate_level_family(builder, n_values)
     (more reliable — observes growth across multiple sizes)

Levels:
    SC(0) - Direct IPS certificate, no selectors needed
    SC(1) - Efficient selectors exist (polynomial circuit cost)
    SC(2) - Selectors exist but are expensive (super-polynomial)
    SC(3?) - Candidate: no useful selectors (certificate search fails)

Author: Carmen Esteban
License: MIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from itertools import combinations
from math import comb, log2
import time

from selector_complexity.patterns import detect_patterns
from selector_complexity.predictor import extract_features, SCPredictor


# =====================================================================
# PHASE 1: STRUCTURAL ANALYSIS
# =====================================================================

def _analyze_structure(axioms, num_vars):
    """Analyze the algebraic and graph structure of the polynomial system."""
    if not axioms:
        return {
            'max_axiom_degree': 0,
            'avg_axiom_degree': 0,
            'var_density': 0.0,
            'connectivity': 0.0,
            'num_linear': 0,
            'num_quadratic': 0,
            'num_higher': 0,
        }

    # Degree distribution
    degrees = []
    for ax in axioms:
        deg = max((len(m) for _, m in ax), default=0)
        degrees.append(deg)

    max_deg = max(degrees)
    avg_deg = sum(degrees) / len(degrees)
    num_linear = sum(1 for d in degrees if d <= 1)
    num_quadratic = sum(1 for d in degrees if d == 2)
    num_higher = sum(1 for d in degrees if d > 2)

    # Variable usage: which variables appear in axioms
    vars_used = set()
    for ax in axioms:
        for _, m in ax:
            vars_used.update(m)
    var_density = len(vars_used) / max(num_vars, 1)

    # Constraint graph: variables are nodes, axioms create edges
    adj = {}
    for ax in axioms:
        ax_vars = set()
        for _, m in ax:
            ax_vars.update(m)
        ax_vars = list(ax_vars)
        for i, v1 in enumerate(ax_vars):
            for v2 in ax_vars[i + 1:]:
                edge = (min(v1, v2), max(v1, v2))
                adj[edge] = adj.get(edge, 0) + 1

    max_edges = num_vars * (num_vars - 1) // 2
    connectivity = len(adj) / max(max_edges, 1)

    return {
        'max_axiom_degree': max_deg,
        'avg_axiom_degree': avg_deg,
        'var_density': var_density,
        'connectivity': connectivity,
        'num_linear': num_linear,
        'num_quadratic': num_quadratic,
        'num_higher': num_higher,
    }


# =====================================================================
# PHASE 2: IPS CERTIFICATE SEARCH
# =====================================================================

def _search_certificates(axioms, num_vars, max_degree, verbose,
                         incremental=True):
    """Search for IPS certificates at increasing degrees.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int
    max_degree : int
    verbose : bool
    incremental : bool
        If True, use IncrementalIPSState to reuse monomial enumeration
        across degrees (default True). Falls back to standard if False.
    """
    if incremental:
        from selector_complexity.solvers import incremental_certificate_search
        return incremental_certificate_search(
            axioms, num_vars, max_degree=max_degree,
            min_degree=2, verbose=verbose)

    # Original non-incremental path
    results = []

    for d in range(2, max_degree + 1):
        num_monoms_est = sum(comb(num_vars, k) for k in range(d + 1))
        if num_monoms_est > 1000000:
            if verbose:
                print("    d={}: ~{} monomials, skipping (too large)".format(
                    d, int(num_monoms_est)))
            break

        t0 = time.time()

        # Build monomial basis
        all_monoms = []
        monom_to_idx = {}
        for deg in range(d + 1):
            for combo in combinations(range(num_vars), deg):
                m = frozenset(combo)
                monom_to_idx[m] = len(all_monoms)
                all_monoms.append(m)
        num_monoms = len(all_monoms)

        # Build IPS matrix
        rows, cols, vals = [], [], []
        total_unknowns = 0

        for ax in axioms:
            deg_ax = max((len(m) for _, m in ax), default=0)
            deg_mult = max(0, d - deg_ax)

            for deg_m in range(deg_mult + 1):
                for combo in combinations(range(num_vars), deg_m):
                    m_mult = frozenset(combo)
                    col = total_unknowns
                    total_unknowns += 1

                    for coef_ax, m_ax in ax:
                        m_prod = m_mult | m_ax
                        if len(m_prod) <= d and m_prod in monom_to_idx:
                            rows.append(monom_to_idx[m_prod])
                            cols.append(col)
                            vals.append(coef_ax)

        if total_unknowns == 0:
            elapsed = time.time() - t0
            results.append({
                'degree': d, 'feasible': False, 'residual': 1.0,
                'size': 0, 'num_monoms': num_monoms,
                'num_unknowns': 0, 'time': elapsed,
            })
            if verbose:
                print("    d={}: no unknowns [{:.2f}s]".format(d, elapsed))
            continue

        A = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(num_monoms, total_unknowns))
        b = np.zeros(num_monoms)
        b[monom_to_idx[frozenset()]] = 1.0

        sol = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=10000)
        x = sol[0]
        residual = float(np.linalg.norm(A @ x - b))
        size = int(np.sum(np.abs(x) > 1e-8))
        elapsed = time.time() - t0

        feasible = residual < 1e-6
        results.append({
            'degree': d, 'feasible': feasible, 'residual': residual,
            'size': size, 'num_monoms': num_monoms,
            'num_unknowns': total_unknowns, 'time': elapsed,
        })

        status = "FEASIBLE" if feasible else "INFEASIBLE"
        if verbose:
            print("    d={}: {} (res={:.2e}, size={}, monoms={}) [{:.2f}s]".format(
                d, status, residual, size, num_monoms, elapsed))

        if feasible:
            break

    return results


# =====================================================================
# PHASE 3: GROWTH ANALYSIS (single instance)
# =====================================================================

def _analyze_growth(certificates, num_vars, structure):
    """Analyze certificate properties relative to system structure."""
    feasible = [c for c in certificates if c['feasible']]

    if not feasible:
        return {
            'pattern': 'no_certificate',
            'min_degree': None,
            'min_size': None,
        }

    best = feasible[0]
    degree = best['degree']
    size = best['size']
    max_axiom_deg = structure.get('max_axiom_degree', 0)

    # Degree gap: how much extra degree beyond axiom structure
    degree_gap = degree - max_axiom_deg

    return {
        'pattern': 'certificate_found',
        'min_degree': degree,
        'min_size': size,
        'degree_gap': degree_gap,
    }


# =====================================================================
# PHASE 4: CLASSIFICATION (single instance)
# =====================================================================

def _classify(evidence):
    """Classify the system into an SC level based on collected evidence."""
    certs = evidence['certificates']
    structure = evidence['structure']
    growth = evidence.get('growth', {})

    feasible_certs = [c for c in certs if c['feasible']]
    max_axiom_deg = structure.get('max_axiom_degree', 0)

    # ── Certificate found ──
    if feasible_certs:
        best = feasible_certs[0]
        degree_gap = best['degree'] - max_axiom_deg

        # SC(0): Certificate exists with small degree gap.
        # The IPS certificate degree is close to the axiom degree,
        # meaning no auxiliary decomposition (selectors) is needed.
        if degree_gap <= 3:
            reasoning = (
                "Certificate found at degree {} (axiom degree {}, gap {}). "
                "Small degree gap indicates a direct IPS proof is possible "
                "without selector decomposition.".format(
                    best['degree'], max_axiom_deg, degree_gap)
            )
            return 0, 'high', reasoning

        # SC(1): Certificate exists but needs larger degree gap.
        # Suggests selector-like decomposition would help.
        if degree_gap <= 6:
            reasoning = (
                "Certificate found at degree {} (axiom degree {}, gap {}). "
                "Moderate degree gap suggests the system benefits from "
                "selector-based case decomposition.".format(
                    best['degree'], max_axiom_deg, degree_gap)
            )
            return 1, 'medium', reasoning

        # SC(2): Certificate exists but at very high degree.
        reasoning = (
            "Certificate found at degree {} (axiom degree {}, gap {}). "
            "Large degree gap suggests expensive selectors are needed.".format(
                best['degree'], max_axiom_deg, degree_gap)
        )
        return 2, 'low', reasoning

    # ── No certificate found ──
    max_tested = max((c['degree'] for c in certs), default=0)
    conn = structure.get('connectivity', 0)

    # Look at residual trend
    residuals = [c['residual'] for c in certs if c['residual'] < 1.0]
    improving = False
    if len(residuals) >= 2:
        improving = residuals[-1] < residuals[0] * 0.5

    if improving:
        # Residuals are dropping — might find certificate at higher degree
        reasoning = (
            "No certificate found up to degree {}, but residuals are "
            "decreasing ({:.2e} -> {:.2e}). May find certificate at "
            "higher degree. Likely SC(1) or SC(2).".format(
                max_tested, residuals[0], residuals[-1])
        )
        return 2, 'low', reasoning

    if conn > 0.3:
        reasoning = (
            "No certificate found up to degree {}. "
            "High constraint connectivity ({:.2f}) suggests "
            "expander-like structure resisting decomposition. "
            "Strong Level 3 candidate.".format(max_tested, conn)
        )
        return '3?', 'medium', reasoning

    reasoning = (
        "No certificate found up to degree {}. "
        "May need higher degree search or different encoding.".format(
            max_tested)
    )
    return '3?', 'low', reasoning


# =====================================================================
# SUMMARY
# =====================================================================

def _build_summary(level, confidence, reasoning, evidence):
    """Build a human-readable summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CLASSIFICATION RESULT")
    lines.append("=" * 60)
    lines.append("")

    level_str = "SC({})".format(level)
    lines.append("  Estimated level:  {}".format(level_str))
    lines.append("  Confidence:       {}".format(confidence))
    lines.append("")
    lines.append("  Reasoning:")
    lines.append("    " + reasoning)
    lines.append("")

    descriptions = {
        0: "No selectors needed. The system has a direct IPS\n"
           "    certificate with degree close to axiom degree.",
        1: "Efficient selectors exist. The system benefits from\n"
           "    case decomposition with polynomial-cost selectors.",
        2: "Selectors exist but are expensive. Any selector family\n"
           "    has super-polynomial (possibly factorial) cost.",
        '3?': "CANDIDATE for Level 3. No useful selectors found.\n"
              "    The system may resist all selector strategies.\n"
              "    (Use estimate_level_family() for stronger evidence.)",
    }
    lines.append("  What this means:")
    lines.append("    " + descriptions.get(level, "Unknown level."))
    lines.append("")

    certs = evidence.get('certificates', [])
    feasible = [c for c in certs if c['feasible']]
    if feasible:
        best = feasible[0]
        lines.append("  Best certificate: degree={}, size={}, residual={:.2e}".format(
            best['degree'], best['size'], best['residual']))
    else:
        max_d = max((c['degree'] for c in certs), default=0)
        lines.append("  No certificate found (tested up to degree {})".format(max_d))

    lines.append("=" * 60)
    return "\n".join(lines)


# =====================================================================
# FAMILY-LEVEL SCALING ANALYSIS
# =====================================================================

def _fit_growth_rate(n_values, sizes):
    """Fit growth rate: constant, polynomial, exponential, or factorial."""
    if len(n_values) < 2:
        return 'insufficient_data', {}

    ns = np.array(n_values, dtype=float)
    ss = np.array(sizes, dtype=float)

    # Constant check: if values barely change, growth is constant (O(1))
    if ss.min() > 0 and (ss.max() - ss.min()) / ss.max() < 0.15:
        details = {
            'poly_degree': 0.0,
            'poly_residual': 0.0,
            'exp_base': 1.0,
            'exp_residual': 0.0,
            'factorial_ratios': [],
            'constant_value': float(ss.mean()),
        }
        return 'constant', details

    # Polynomial fit: log(size) ~ k * log(n)
    log_ns = np.log(ns + 1e-10)
    log_ss = np.log(ss + 1e-10)

    if len(ns) >= 2:
        poly_fit = np.polyfit(log_ns, log_ss, 1)
        poly_degree = poly_fit[0]
        poly_residual = np.mean((np.polyval(poly_fit, log_ns) - log_ss) ** 2)
    else:
        poly_degree = 0
        poly_residual = float('inf')

    # Exponential fit: log(size) ~ k * n
    if len(ns) >= 2:
        exp_fit = np.polyfit(ns, log_ss, 1)
        exp_base = np.exp(exp_fit[0])
        exp_residual = np.mean((np.polyval(exp_fit, ns) - log_ss) ** 2)
    else:
        exp_base = 0
        exp_residual = float('inf')

    # Factorial check: compare sizes to n!
    factorial_ratios = []
    for n_val, s in zip(n_values, sizes):
        n_fact = 1
        for i in range(2, int(n_val) + 1):
            n_fact *= i
        if n_fact > 0:
            factorial_ratios.append(s / n_fact)

    details = {
        'poly_degree': float(poly_degree),
        'poly_residual': float(poly_residual),
        'exp_base': float(exp_base),
        'exp_residual': float(exp_residual),
        'factorial_ratios': factorial_ratios,
    }

    # Classification logic
    if poly_residual < exp_residual and poly_degree < 6:
        return 'polynomial', details
    elif factorial_ratios and all(0.01 < r < 100 for r in factorial_ratios):
        return 'factorial', details
    elif exp_residual < poly_residual:
        return 'exponential', details
    else:
        return 'super-polynomial', details


def estimate_level_family(builder, n_values, max_degree=8, verbose=True):
    """Estimate SC level by analyzing a family across multiple sizes.

    This is more reliable than single-instance classification because
    it observes the actual growth rate of certificate size/degree.

    Parameters
    ----------
    builder : callable
        Function n -> (axioms, num_vars) or n -> (axioms, num_vars, ...).
        Must return axioms as first element and num_vars as second.
    n_values : list of int
        Parameter values to test (e.g., [2, 3, 4, 5]).
    max_degree : int, optional
        Maximum IPS certificate degree to search per instance.
    verbose : bool, optional
        Print progress information.

    Returns
    -------
    dict
        'level' : int or str
        'confidence' : str
        'scaling' : dict with growth rate analysis
        'instances' : list of per-instance results
        'summary' : str
    """
    if verbose:
        print("=" * 60)
        print("SC-LEVEL FAMILY CLASSIFIER")
        print("=" * 60)
        print("  Testing n = {}".format(list(n_values)))
        print()

    instances = []

    for n in n_values:
        if verbose:
            print("  --- n = {} ---".format(n))

        result = builder(n)
        axioms = result[0]
        num_vars = result[1]

        structure = _analyze_structure(axioms, num_vars)
        certs = _search_certificates(axioms, num_vars, max_degree, verbose)

        feasible = [c for c in certs if c['feasible']]
        best = feasible[0] if feasible else None

        instances.append({
            'n': n,
            'num_vars': num_vars,
            'num_axioms': len(axioms),
            'max_axiom_degree': structure['max_axiom_degree'],
            'best_cert': best,
            'all_certs': certs,
        })

        if verbose and best:
            print("    => degree={}, size={}".format(
                best['degree'], best['size']))
        elif verbose:
            print("    => NO CERTIFICATE FOUND")
        if verbose:
            print()

    # Scaling analysis
    if verbose:
        print("  Scaling analysis:")
        print("  " + "-" * 50)

    feasible_instances = [inst for inst in instances if inst['best_cert']]
    infeasible_instances = [inst for inst in instances if not inst['best_cert']]

    if len(feasible_instances) >= 2:
        # Analyze degree growth
        ns_d = [inst['n'] for inst in feasible_instances]
        degrees = [inst['best_cert']['degree'] for inst in feasible_instances]
        sizes = [inst['best_cert']['size'] for inst in feasible_instances]

        degree_growth, d_details = _fit_growth_rate(ns_d, degrees)
        size_growth, s_details = _fit_growth_rate(ns_d, sizes)

        if verbose:
            print("    {:>4} | {:>8} | {:>10} | {:>12}".format(
                "n", "vars", "cert_deg", "cert_size"))
            print("    " + "-" * 40)
            for inst in feasible_instances:
                b = inst['best_cert']
                print("    {:>4} | {:>8} | {:>10} | {:>12}".format(
                    inst['n'], inst['num_vars'], b['degree'], b['size']))
            print()
            print("    Degree growth: {} (poly_degree={:.1f})".format(
                degree_growth, d_details.get('poly_degree', 0)))
            print("    Size growth:   {} (poly_degree={:.1f})".format(
                size_growth, s_details.get('poly_degree', 0)))

        scaling = {
            'degree_growth': degree_growth,
            'degree_details': d_details,
            'size_growth': size_growth,
            'size_details': s_details,
        }

    else:
        scaling = {
            'degree_growth': 'no_certificate' if not feasible_instances else 'single_point',
            'size_growth': 'no_certificate' if not feasible_instances else 'single_point',
        }

    # Family-level classification
    level, confidence, reasoning = _classify_family(
        instances, feasible_instances, infeasible_instances, scaling)

    summary = _build_family_summary(
        level, confidence, reasoning, instances, scaling)

    if verbose:
        print()
        print(summary)

    return {
        'level': level,
        'confidence': confidence,
        'scaling': scaling,
        'instances': instances,
        'summary': summary,
    }


def _classify_family(instances, feasible, infeasible, scaling):
    """Classify based on family scaling behavior.

    Primary signal: certificate DEGREE growth (reliable).
    Secondary signal: certificate SIZE growth (LSQR upper bound, less reliable).

    The LSQR solver minimizes L2 norm, not sparsity, so certificate sizes
    from LSQR are upper bounds on the true optimal size. The DEGREE at
    which the system first becomes feasible is the most reliable signal.
    """

    # All infeasible → SC(3?) candidate
    if not feasible:
        reasoning = (
            "No IPS certificate found for ANY tested instance. "
            "Strong candidate for Level 3 (no useful selectors)."
        )
        return '3?', 'high', reasoning

    # Some feasible, some not → transitioning, likely SC(2) or SC(3?)
    if infeasible and feasible:
        max_feasible_n = max(inst['n'] for inst in feasible)
        min_infeasible_n = min(inst['n'] for inst in infeasible)
        reasoning = (
            "Certificate found for n <= {} but NOT for n >= {}. "
            "The system becomes infeasible as n grows (within tested "
            "degree range), suggesting high certificate complexity.".format(
                max_feasible_n, min_infeasible_n)
        )
        return 2, 'medium', reasoning

    # All feasible → analyze DEGREE growth (primary signal)
    degree_growth = scaling.get('degree_growth', 'unknown')
    d_details = scaling.get('degree_details', {})

    # Check degree gaps
    degree_gaps = []
    for inst in feasible:
        gap = inst['best_cert']['degree'] - inst['max_axiom_degree']
        degree_gaps.append(gap)
    avg_gap = sum(degree_gaps) / len(degree_gaps) if degree_gaps else 0
    max_gap = max(degree_gaps) if degree_gaps else 0

    # SC(0): Degree is constant or grows polynomially with n.
    # Constant degree is the strongest SC(0) signal — bounded-degree
    # certificates exist for all n regardless of system size.
    if degree_growth == 'constant':
        const_val = d_details.get('constant_value', 0)
        reasoning = (
            "Certificate degree is constant (~{:.0f}) across all tested n. "
            "Bounded-degree IPS certificates exist — "
            "no selectors needed.".format(const_val)
        )
        return 0, 'high', reasoning

    if degree_growth == 'polynomial':
        poly_deg = d_details.get('poly_degree', 0)
        if poly_deg <= 2 and max_gap <= 4:
            reasoning = (
                "Certificate degree grows polynomially "
                "(degree ~ n^{:.1f}), with average degree gap {:.1f}. "
                "This indicates direct IPS certificates — "
                "no selectors needed.".format(poly_deg, avg_gap)
            )
            return 0, 'high', reasoning

        # Polynomial degree growth but larger gaps → may need selectors
        reasoning = (
            "Certificate degree grows polynomially "
            "(degree ~ n^{:.1f}), but degree gap up to {}. "
            "Likely benefits from selector decomposition.".format(
                poly_deg, max_gap)
        )
        return 1, 'medium', reasoning

    # SC(1) or SC(2): super-polynomial degree growth
    if degree_growth in ('exponential', 'factorial'):
        reasoning = (
            "Certificate degree grows {} with n. "
            "Selectors exist but may be expensive.".format(degree_growth)
        )
        return 2, 'medium', reasoning

    if degree_growth == 'super-polynomial':
        reasoning = (
            "Certificate degree grows super-polynomially. "
            "Likely SC(1) or SC(2) depending on exact growth rate."
        )
        return 1, 'low', reasoning

    # Fallback
    reasoning = (
        "Certificate found for all instances. "
        "Degree growth pattern: {}. "
        "Average degree gap: {:.1f}.".format(degree_growth, avg_gap)
    )
    return 0, 'low', reasoning


def _build_family_summary(level, confidence, reasoning, instances, scaling):
    """Build summary for family-level classification."""
    lines = []
    lines.append("=" * 60)
    lines.append("  FAMILY CLASSIFICATION RESULT")
    lines.append("=" * 60)
    lines.append("")

    level_str = "SC({})".format(level)
    lines.append("  Estimated level:  {}".format(level_str))
    lines.append("  Confidence:       {}".format(confidence))
    lines.append("")
    lines.append("  Reasoning:")
    lines.append("    " + reasoning)
    lines.append("")

    # Instance table
    lines.append("  Instance details:")
    lines.append("    {:>4} | {:>6} | {:>8} | {:>10} | {:>10}".format(
        "n", "vars", "axioms", "cert_deg", "cert_size"))
    lines.append("    " + "-" * 48)

    for inst in instances:
        b = inst['best_cert']
        if b:
            lines.append("    {:>4} | {:>6} | {:>8} | {:>10} | {:>10}".format(
                inst['n'], inst['num_vars'], inst['num_axioms'],
                b['degree'], b['size']))
        else:
            lines.append("    {:>4} | {:>6} | {:>8} | {:>10} | {:>10}".format(
                inst['n'], inst['num_vars'], inst['num_axioms'],
                "NONE", "-"))

    lines.append("")

    descriptions = {
        0: "SC(0): No selectors needed — direct polynomial IPS proof.",
        1: "SC(1): Efficient selectors exist — polynomial cost decomposition.",
        2: "SC(2): Selectors exist but expensive — super-polynomial cost.",
        '3?': "SC(3?): CANDIDATE — no useful selectors at tested degrees.",
    }
    lines.append("  " + descriptions.get(level, ""))
    lines.append("=" * 60)
    return "\n".join(lines)


# =====================================================================
# PUBLIC API: SINGLE INSTANCE
# =====================================================================

def estimate_level(axioms, num_vars, max_degree=6, verbose=True,
                   use_predictor=True, use_patterns=True, incremental=True):
    """Estimate the Selector Complexity level of a polynomial system.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        The polynomial axioms encoding the tautology.
        Each axiom is a list of (coefficient, monomial) pairs where
        the monomial is a frozenset of variable indices.
    num_vars : int
        Number of variables in the system.
    max_degree : int, optional
        Maximum IPS certificate degree to search (default 6).
    verbose : bool, optional
        Print progress and diagnostic information (default True).
    use_predictor : bool, optional
        Use SCPredictor for fast estimation (default True).
        If confidence > 0.8, returns without running LSQR.
    use_patterns : bool, optional
        Use pattern detection shortcuts (default True).
        If a pattern matches with high confidence, returns without LSQR.
    incremental : bool, optional
        Use incremental matrix construction (default True).

    Returns
    -------
    dict
        'level' : int or str
            Estimated SC level (0, 1, 2, or '3?').
        'confidence' : str
            'high', 'medium', or 'low'.
        'evidence' : dict
            Detailed test results from each phase.
        'summary' : str
            Human-readable classification report.

    Examples
    --------
    >>> from selector_complexity import php_axioms, estimate_level
    >>> axioms, nv, _ = php_axioms(3)
    >>> result = estimate_level(axioms, nv, verbose=False)
    >>> result['level']
    0
    """
    if verbose:
        print("=" * 60)
        print("SC-LEVEL CLASSIFIER")
        print("=" * 60)
        print("  System: {} axioms, {} variables".format(
            len(axioms), num_vars))
        print()

    evidence = {
        'num_axioms': len(axioms),
        'num_vars': num_vars,
        'certificates': [],
        'structure': {},
    }

    # Phase 1: Structural analysis
    if verbose:
        print("  Phase 1: Structural analysis...")

    structure = _analyze_structure(axioms, num_vars)
    evidence['structure'] = structure

    if verbose:
        print("    Max axiom degree: {}".format(structure['max_axiom_degree']))
        print("    Variable density: {:.2f}".format(structure['var_density']))
        print("    Connectivity: {:.2f}".format(structure['connectivity']))
        print()

    # Phase 1.5: Pattern detection shortcuts
    if use_patterns:
        if verbose:
            print("  Phase 1.5: Pattern detection...")

        pattern_result = detect_patterns(axioms, num_vars)
        evidence['patterns'] = pattern_result

        if (pattern_result['shortcut_available']
                and pattern_result['shortcut_confidence'] == 'high'):
            level = pattern_result['shortcut_level']
            confidence = 'high'
            reasoning = (
                "Pattern shortcut: {} detected with high confidence. "
                "Estimated SC({}) without IPS search.".format(
                    pattern_result['shortcut_source'], level))
            summary = _build_summary(level, confidence, reasoning, evidence)

            if verbose:
                print("    Shortcut: {} -> SC({})".format(
                    pattern_result['shortcut_source'], level))
                print()
                print(summary)

            return {
                'level': level,
                'confidence': confidence,
                'evidence': evidence,
                'summary': summary,
            }

        if verbose:
            if pattern_result['shortcut_available']:
                print("    Pattern found: {} (confidence: {})".format(
                    pattern_result['shortcut_source'],
                    pattern_result['shortcut_confidence']))
            else:
                print("    No high-confidence pattern shortcut.")
            print()

    # Phase 1.7: Predictor shortcut
    if use_predictor:
        if verbose:
            print("  Phase 1.7: SC Predictor...")

        try:
            features = extract_features(axioms, num_vars)
            evidence['features'] = features

            predictor = SCPredictor()
            predictor.fit_from_landscape()
            pred_result = predictor.predict(features)
            evidence['predictor'] = pred_result

            if pred_result['confidence'] > 0.8:
                level = pred_result['predicted_level']
                confidence = 'medium'  # predictor alone is medium confidence
                reasoning = (
                    "Predictor shortcut: SC({}) predicted with {:.0%} confidence. "
                    "{}".format(level, pred_result['confidence'],
                                pred_result['reasoning']))
                summary = _build_summary(level, confidence, reasoning, evidence)

                if verbose:
                    print("    Predicted: SC({}) (confidence: {:.0%})".format(
                        level, pred_result['confidence']))
                    print("    Using predictor shortcut.")
                    print()
                    print(summary)

                return {
                    'level': level,
                    'confidence': confidence,
                    'evidence': evidence,
                    'summary': summary,
                }

            if verbose:
                print("    Predicted: SC({}) (confidence: {:.0%}, below threshold)".format(
                    pred_result['predicted_level'], pred_result['confidence']))
                print()
        except Exception:
            if verbose:
                print("    Predictor unavailable, continuing with LSQR...")
                print()

    # Phase 2: IPS certificate search
    if verbose:
        print("  Phase 2: IPS certificate search...")

    certificates = _search_certificates(axioms, num_vars, max_degree, verbose,
                                         incremental=incremental)
    evidence['certificates'] = certificates

    # Phase 3: Growth analysis
    if verbose:
        print()
        print("  Phase 3: Growth analysis...")

    growth = _analyze_growth(certificates, num_vars, structure)
    evidence['growth'] = growth

    if verbose:
        print("    Pattern: {}".format(growth['pattern']))
        if growth.get('degree_gap') is not None:
            print("    Degree gap (cert - axiom): {}".format(
                growth['degree_gap']))

    # Phase 4: Classification
    if verbose:
        print()
        print("  Phase 4: Classification...")

    level, confidence, reasoning = _classify(evidence)
    summary = _build_summary(level, confidence, reasoning, evidence)

    if verbose:
        print()
        print(summary)

    return {
        'level': level,
        'confidence': confidence,
        'evidence': evidence,
        'summary': summary,
    }
