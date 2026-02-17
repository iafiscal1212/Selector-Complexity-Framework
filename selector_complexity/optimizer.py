"""
Optimized Certificate Search
=============================

Predicts minimum certificate degree and accelerates IPS search
by skipping degrees that are clearly insufficient.

Functions:
    predict_min_degree           -- heuristic degree prediction
    optimized_certificate_search -- accelerated search with plateau detection
    parallel_certificate_search  -- multiprocessing search across degrees

Author: Carmen Esteban
License: MIT
"""

import time

import numpy as np
from scipy.sparse.linalg import lsqr
from math import comb

from selector_complexity.classifier import _analyze_structure
from selector_complexity.solvers import build_matrix, build_matrix_tuples


def predict_min_degree(axioms, num_vars):
    """Predict the minimum IPS certificate degree without building matrices.

    Uses heuristics based on axiom degree, variable connectivity, and
    density to estimate the minimum degree at which a certificate exists.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        The polynomial axioms.
    num_vars : int
        Number of variables.

    Returns
    -------
    dict
        'predicted_min_degree' : int
            Estimated minimum degree.
        'confidence' : str
            'high', 'medium', or 'low'.
        'reasoning' : str
            Explanation of the prediction.
    """
    structure = _analyze_structure(axioms, num_vars)

    max_deg = structure['max_axiom_degree']
    connectivity = structure['connectivity']
    var_density = structure['var_density']
    num_axioms = len(axioms)

    # Base: certificate degree >= max axiom degree
    base_degree = max(2, max_deg)

    # Connectivity bonus: high connectivity -> harder -> higher degree
    conn_bonus = 0
    if connectivity > 0.5:
        conn_bonus = 2
    elif connectivity > 0.3:
        conn_bonus = 1

    # Density bonus: more variables used -> harder
    density_bonus = 0
    if var_density > 0.8:
        density_bonus = 1

    # Axiom ratio: many axioms relative to vars -> more constrained
    ratio = num_axioms / max(num_vars, 1)
    ratio_bonus = 0
    if ratio > 5:
        ratio_bonus = 2
    elif ratio > 2:
        ratio_bonus = 1

    predicted = base_degree + conn_bonus + density_bonus + ratio_bonus

    # Confidence
    if connectivity < 0.2 and var_density < 0.5:
        confidence = 'high'
    elif connectivity > 0.5:
        confidence = 'low'
    else:
        confidence = 'medium'

    reasoning = (
        "Base degree {} (from axiom degree {}). "
        "Adjustments: connectivity {:.2f} (+{}), "
        "density {:.2f} (+{}), axiom ratio {:.1f} (+{}). "
        "Predicted minimum: {}.".format(
            base_degree, max_deg,
            connectivity, conn_bonus,
            var_density, density_bonus,
            ratio, ratio_bonus,
            predicted)
    )

    return {
        'predicted_min_degree': predicted,
        'confidence': confidence,
        'reasoning': reasoning,
    }


def optimized_certificate_search(axioms, num_vars, max_degree=10):
    """Search for IPS certificates with degree skipping and plateau detection.

    Improvements over naive search:
    1. Starts at the predicted minimum degree (skips low degrees)
    2. Detects residual plateaus: if 3 consecutive degrees show no
       improvement (< 10%), stops early to save time

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
        The polynomial axioms.
    num_vars : int
        Number of variables.
    max_degree : int, optional
        Maximum degree to search (default 10).

    Returns
    -------
    dict
        'found' : bool
            Whether a certificate was found.
        'certificate' : dict or None
            Certificate details if found.
        'degrees_searched' : list of int
            Degrees actually searched.
        'degrees_skipped' : list of int
            Degrees skipped by the optimizer.
        'time_saved_estimate' : float
            Estimated time saved vs. naive search (seconds).
        'plateau_detected' : bool
            Whether search was stopped by plateau detection.
        'residuals' : dict
            Mapping degree -> residual for all searched degrees.
        'total_time' : float
            Total elapsed time (seconds).
    """
    t0 = time.time()

    # Predict starting degree
    prediction = predict_min_degree(axioms, num_vars)
    start_degree = max(2, prediction['predicted_min_degree'] - 1)
    start_degree = min(start_degree, max_degree)

    degrees_skipped = list(range(2, start_degree))
    degrees_searched = []
    residuals = {}
    recent_residuals = []
    plateau_detected = False
    certificate = None

    # Estimate time saved for skipped degrees
    time_saved = 0.0
    for d in degrees_skipped:
        num_monoms_est = sum(comb(num_vars, k) for k in range(d + 1))
        time_saved += num_monoms_est * 1e-6  # rough estimate

    for d in range(start_degree, max_degree + 1):
        # Check monomial count limit
        num_monoms_est = sum(comb(num_vars, k) for k in range(d + 1))
        if num_monoms_est > 500000:
            break

        A, b, nm, nu = build_matrix(axioms, num_vars, d)
        if nu == 0:
            degrees_searched.append(d)
            residuals[d] = 1.0
            recent_residuals.append(1.0)
            continue

        sol = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=10000)
        x = sol[0]
        residual = float(np.linalg.norm(A @ x - b))

        degrees_searched.append(d)
        residuals[d] = residual

        if residual < 1e-6:
            # Certificate found
            size = int(np.sum(np.abs(x) > 1e-8))
            certificate = {
                'degree': d,
                'size': size,
                'num_monoms': nm,
                'num_unknowns': nu,
                'residual': residual,
            }
            break

        # Plateau detection: 3 consecutive non-improving residuals
        recent_residuals.append(residual)
        if len(recent_residuals) >= 3:
            last3 = recent_residuals[-3:]
            # Check if improvement < 10% across last 3
            if last3[0] > 0 and (last3[0] - last3[-1]) / last3[0] < 0.10:
                plateau_detected = True
                break

    total_time = time.time() - t0

    return {
        'found': certificate is not None,
        'certificate': certificate,
        'degrees_searched': degrees_searched,
        'degrees_skipped': degrees_skipped,
        'time_saved_estimate': round(time_saved, 4),
        'plateau_detected': plateau_detected,
        'residuals': residuals,
        'total_time': round(total_time, 4),
    }


# =====================================================================
# OPT 2: PARALLEL CERTIFICATE SEARCH
# =====================================================================

def _search_single_degree(args):
    """Worker function for parallel search. Must be at module level for pickle.

    Parameters
    ----------
    args : tuple
        (axioms, num_vars, degree, atol)

    Returns
    -------
    dict with degree result
    """
    axioms, num_vars, d, atol = args
    t0 = time.time()

    num_monoms_est = sum(comb(num_vars, k) for k in range(d + 1))
    if num_monoms_est > 500000:
        return {
            'degree': d, 'feasible': False, 'residual': 1.0,
            'size': 0, 'num_monoms': int(num_monoms_est),
            'num_unknowns': 0, 'time': 0.0, 'skipped': True,
        }

    A, b, nm, nu = build_matrix_tuples(axioms, num_vars, d)

    if nu == 0:
        elapsed = time.time() - t0
        return {
            'degree': d, 'feasible': False, 'residual': 1.0,
            'size': 0, 'num_monoms': nm,
            'num_unknowns': 0, 'time': elapsed, 'skipped': False,
        }

    sol = lsqr(A, b, atol=atol, btol=atol, iter_lim=10000)
    x = sol[0]
    residual = float(np.linalg.norm(A @ x - b))
    size = int(np.sum(np.abs(x) > 1e-8))
    elapsed = time.time() - t0

    return {
        'degree': d, 'feasible': residual < 1e-6, 'residual': residual,
        'size': size, 'num_monoms': nm,
        'num_unknowns': nu, 'time': elapsed, 'skipped': False,
    }


def parallel_certificate_search(axioms, num_vars, max_degree=10,
                                 min_degree=2, max_workers=None, atol=1e-12):
    """Search for IPS certificates in parallel across degrees.

    Each degree is searched as an independent task using ProcessPoolExecutor.
    Useful when you want to test multiple degrees simultaneously.

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int
    max_degree : int
    min_degree : int
    max_workers : int or None
        Number of parallel workers. None = os.cpu_count().
    atol : float

    Returns
    -------
    dict
        'found' : bool
        'certificate' : dict or None (best/lowest degree certificate)
        'all_results' : list of dict (per-degree results)
        'total_time' : float
    """
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    degrees = list(range(min_degree, max_degree + 1))

    tasks = [(axioms, num_vars, d, atol) for d in degrees]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_search_single_degree, tasks):
            results.append(result)

    # Sort by degree and find best certificate
    results.sort(key=lambda r: r['degree'])
    certificate = None
    for r in results:
        if r['feasible']:
            certificate = r
            break

    total_time = time.time() - t0

    return {
        'found': certificate is not None,
        'certificate': certificate,
        'all_results': results,
        'total_time': round(total_time, 4),
    }
