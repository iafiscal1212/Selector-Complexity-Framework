"""
Selector Discovery Engine
==========================

Automatic discovery of efficient selector families for polynomial systems.

Given an unsatisfiable polynomial system (tautology), systematically searches
for selector families that reduce IPS certificate complexity.

Classes:
    DiscoveryConfig   - Search parameters
    SelectorCandidate - A candidate selector family (pre-verification)
    DiscoveryResult   - Complete result, serializable to JSON
    SelectorVerifier  - Exhaustive or algebraic verification
    QualityMeasurer   - Compares cert size with/without selectors
    DiscoveryEngine   - Orchestrates strategies -> verify -> measure -> classify

Author: Carmen Esteban
"""

import json
import time
import numpy as np
from itertools import combinations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from math import comb

from selector_complexity.core import PolynomialSystem, SelectorFamily
from selector_complexity.solvers import build_matrix, find_certificate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryConfig:
    """Parameters controlling the discovery search."""
    max_degree: int = 4
    max_selector_count: int = 20
    max_selector_degree: int = 4
    strategies: List[str] = field(default_factory=lambda: [
        "exhaustive", "template", "ips_guided",
        "subspace_projection", "variable_group", "axiom_graph"
    ])
    max_vars_exhaustive: int = 15
    top_k_variables: int = 10
    monomial_cap: int = 500_000
    use_aip: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# Candidate & Result
# ---------------------------------------------------------------------------

@dataclass
class SelectorCandidate:
    """A candidate selector family before full verification.

    Attributes:
        selectors: dict mapping label -> list of (coef, frozenset) terms
        source: string identifying the strategy that produced this candidate
    """
    selectors: Dict[Any, list]
    source: str
    degree: int = 0
    num_selectors: int = 0
    total_size: int = 0
    pre_verified: bool = False

    def __post_init__(self):
        if self.selectors:
            self.num_selectors = len(self.selectors)
            self.total_size = sum(len(terms) for terms in self.selectors.values())
            self.degree = max(
                max((len(m) for _, m in terms), default=0)
                for terms in self.selectors.values()
            )


@dataclass
class DiscoveryResult:
    """Complete result of selector discovery, serializable to JSON."""
    system: str
    n: int
    level_estimate: int
    candidates_tested: int
    candidates_verified: int
    best_selector: Optional[Dict] = None
    cert_without_selectors: Optional[Dict] = None
    cert_with_selectors: Optional[Dict] = None
    all_candidates: List[Dict] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def to_json(self):
        d = asdict(self)
        # Remove large coefficient arrays from serialization
        for key in ("cert_without_selectors", "cert_with_selectors"):
            if d.get(key) and "coefficients" in d[key]:
                del d[key]["coefficients"]
        return json.dumps(d, indent=2, default=str)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# Selector serialization
# ---------------------------------------------------------------------------

def _serialize_selectors(selectors):
    """Serialize selector polynomials to JSON-safe format.

    Converts {label: [(coef, frozenset), ...]} to
    {str(label): [{"c": coef, "m": sorted_list}, ...]}.
    """
    out = {}
    for label, terms in selectors.items():
        out[str(label)] = [
            {"c": round(c, 10), "m": sorted(m)}
            for c, m in terms if abs(c) > 1e-12
        ]
    return out


def _deserialize_selectors(data):
    """Inverse of _serialize_selectors. Returns dict of label -> [(coef, frozenset)]."""
    out = {}
    for label, terms in data.items():
        out[label] = [(t["c"], frozenset(t["m"])) for t in terms]
    return out


# ---------------------------------------------------------------------------
# Solver wrapper (aip-engine integration)
# ---------------------------------------------------------------------------

def _solve_with_aip(A, b, use_aip=True):
    """Route to aip.solve() when available, fall back to scipy LSQR."""
    if use_aip:
        try:
            import sys
            sys.path.insert(0, "/home/caresment/aip-engine-github/src")
            import aip
            return aip.solve(A, b)
        except ImportError:
            pass
    from scipy.sparse.linalg import lsqr
    result = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=10000)
    return result[0]


def _solve_large_with_accordion(axioms, num_vars, d_max, use_aip=True):
    """Use AccordionBuilder + solve_chunks for large systems (40+ vars)."""
    if not use_aip:
        return find_certificate(axioms, num_vars, d_max)
    try:
        import sys
        sys.path.insert(0, "/home/caresment/aip-engine-github/src")
        from aip.accordion import AccordionBuilder, solve_chunks, PascalIndex

        pidx = PascalIndex(num_vars, d_max)
        num_monoms = pidx.total_monomials()
        builder = AccordionBuilder(num_monoms)

        col = 0
        for ax in axioms:
            deg_ax = max(len(m) for c, m in ax)
            deg_mult = max(0, d_max - deg_ax)
            for d in range(deg_mult + 1):
                for combo in combinations(range(num_vars), d):
                    m_mult = frozenset(combo)
                    for coef_ax, m_ax in ax:
                        m_prod = m_mult | m_ax
                        if len(m_prod) <= d_max:
                            row = pidx.combo_to_index(tuple(sorted(m_prod)))
                            builder.add_entry(row, col, coef_ax)
                    col += 1

        chunks = builder.finalize()
        b = np.zeros(num_monoms)
        b[0] = 1.0  # constant monomial = 1

        result = solve_chunks(chunks, b, verbose=False)
        if result["feasible"]:
            return {
                "degree": d_max,
                "size": int(np.sum(np.abs(result["x"]) > 1e-8)),
                "num_monoms": num_monoms,
                "num_unknowns": col,
                "residual": float(result["residual"]),
                "coefficients": result["x"],
            }
        return None
    except (ImportError, Exception):
        return find_certificate(axioms, num_vars, d_max)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class SelectorVerifier:
    """Verify selector candidates rigorously.

    Uses exhaustive boolean-cube evaluation for small systems (n <= threshold)
    and algebraic + spot-check verification for larger ones.
    """

    def __init__(self, num_vars, max_vars_exhaustive=15):
        self.num_vars = num_vars
        self.max_vars_exhaustive = max_vars_exhaustive

    def verify(self, candidate):
        """Full verification: partition of unity + mutual exclusivity."""
        if self.num_vars <= self.max_vars_exhaustive:
            return self._verify_exhaustive(candidate)
        return self._verify_algebraic(candidate)

    def _verify_exhaustive(self, candidate):
        """Check on full boolean cube {0,1}^n."""
        labels = list(candidate.selectors.keys())

        for bits in range(2 ** self.num_vars):
            assignment = {v: (bits >> v) & 1 for v in range(self.num_vars)}
            vals = {}
            for label in labels:
                v = 0.0
                for coef, monom in candidate.selectors[label]:
                    prod = coef
                    for var in monom:
                        prod *= assignment.get(var, 0)
                    v += prod
                vals[label] = v

            # Partition of unity: sum = 1
            if abs(sum(vals.values()) - 1.0) > 1e-8:
                return False
            # Boolean values: each selector is 0 or 1
            if not all(abs(v) < 1e-8 or abs(v - 1) < 1e-8 for v in vals.values()):
                return False
            # Mutual exclusivity: g_i * g_j = 0 for i != j
            for i, l1 in enumerate(labels):
                for l2 in labels[i + 1:]:
                    if abs(vals[l1] * vals[l2]) > 1e-8:
                        return False
        return True

    def _verify_algebraic(self, candidate):
        """Algebraic verification for larger systems.

        Symbolically checks partition of unity (sum of selectors = 1),
        then spot-checks exclusivity on random boolean assignments.
        """
        labels = list(candidate.selectors.keys())

        # Symbolic partition check: sum all coefficients per monomial
        combined = {}
        for label in labels:
            for coef, monom in candidate.selectors[label]:
                combined[monom] = combined.get(monom, 0.0) + coef

        # Constant term should be 1, all others 0
        if abs(combined.get(frozenset(), 0.0) - 1.0) > 1e-8:
            return False
        for monom, coef in combined.items():
            if monom != frozenset() and abs(coef) > 1e-8:
                return False

        # Spot-check exclusivity on random assignments
        rng = np.random.RandomState(42)
        for _ in range(1000):
            assignment = {v: rng.randint(0, 2) for v in range(self.num_vars)}
            vals = {}
            for label in labels:
                v = 0.0
                for coef, monom in candidate.selectors[label]:
                    prod = coef
                    for var in monom:
                        prod *= assignment.get(var, 0)
                    v += prod
                vals[label] = v
            for i, l1 in enumerate(labels):
                for l2 in labels[i + 1:]:
                    if abs(vals[l1] * vals[l2]) > 1e-8:
                        return False
        return True


# ---------------------------------------------------------------------------
# Quality measurement
# ---------------------------------------------------------------------------

class QualityMeasurer:
    """Measure selector benefit using IPS-with-selectors certificates.

    IPS-with-selectors: 1 = sum_i g_i(x) * C_i(x)
    where C_i = sum_j c_{i,j}(x) * f_j(x) and the total degree of
    each term g_i * c_{i,j} * f_j is at most D.

    The key is that branch multipliers c_{i,j} have degree
    <= D - deg(g_i) - deg(f_j), which is LESS than without selectors
    (where multipliers have degree <= D - deg(f_j)). With k branches
    of reduced-degree multipliers, the total cert can be smaller.
    """

    def __init__(self, config):
        self.config = config

    def measure(self, axioms, num_vars, candidate, baseline_cert):
        """Find IPS-with-selectors certificate at increasing degree.

        Builds a single linear system: 1 = sum_i g_i * (sum_j c_{i,j} * f_j)
        where unknowns are all the c_{i,j} multiplier coefficients.
        """
        labels = list(candidate.selectors.keys())

        for d in range(1, self.config.max_degree + 1):
            total_monoms = sum(comb(num_vars, k) for k in range(d + 1))
            if total_monoms > self.config.monomial_cap:
                break

            result = self._try_selector_cert(axioms, num_vars, d,
                                             candidate, labels)
            if result is not None:
                return result

        return None

    def _try_selector_cert(self, axioms, num_vars, D, candidate, labels):
        """Build and solve the IPS-with-selectors system at total degree D.

        System: 1 = sum_i g_i * (sum_j c_{i,j} * f_j)
        Each g_i * c_{i,j} * f_j has degree <= D.
        Unknowns: coefficients of each c_{i,j} (one per branch per axiom
        per multiplier monomial).
        """
        from scipy import sparse
        from scipy.sparse.linalg import lsqr

        # Build monomial indexing
        all_monoms = []
        monom_to_idx = {}
        for d in range(D + 1):
            for combo in combinations(range(num_vars), d):
                m = frozenset(combo)
                monom_to_idx[m] = len(all_monoms)
                all_monoms.append(m)
        num_monoms = len(all_monoms)

        # Build the combined system matrix
        rows, cols, vals = [], [], []
        total_unknowns = 0

        for label in labels:
            g_i = candidate.selectors[label]
            d_g = max((len(m) for _, m in g_i), default=0)

            for ax in axioms:
                d_ax = max(len(m) for c, m in ax)
                d_mult = max(0, D - d_g - d_ax)

                for dm in range(d_mult + 1):
                    for combo in combinations(range(num_vars), dm):
                        m_mult = frozenset(combo)
                        col = total_unknowns
                        total_unknowns += 1

                        # Contribution: g_i * m_mult * f_j
                        for c_g, m_g in g_i:
                            for c_ax, m_ax in ax:
                                m_prod = m_g | m_mult | m_ax
                                if (len(m_prod) <= D and
                                        m_prod in monom_to_idx):
                                    rows.append(monom_to_idx[m_prod])
                                    cols.append(col)
                                    vals.append(c_g * c_ax)

        if total_unknowns == 0:
            return None

        A = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(num_monoms, total_unknowns))

        b = np.zeros(num_monoms)
        b[monom_to_idx[frozenset()]] = 1.0

        res = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=10000)
        x = res[0]
        residual = np.linalg.norm(A @ x - b)

        if residual > 1e-6:
            return None

        size = int(np.sum(np.abs(x) > 1e-8))
        return {
            "degree": D,
            "size": size,
            "num_monoms": num_monoms,
            "num_unknowns": total_unknowns,
            "residual": float(residual),
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_level(baseline_cert, best_candidate, cert_with_selectors,
                    config, var_maps=None):
    """Classify selector complexity level.

    Uses both the existence of auxiliary variables and whether selectors
    provide genuine certificate size reduction (branch certs < baseline).

    Returns:
        0 - Direct polynomial certificate, no selectors needed (e.g. PHP)
        1 - Efficient selectors reduce certificate size (e.g. PHP-E)
        2 - Selectors found but don't reduce cert size (e.g. PHP-C)
        3 - No useful selectors found (e.g. Tseitin)
    """
    has_auxiliary = (var_maps is not None and
                     any(k != "var_x" for k in var_maps))

    if has_auxiliary and best_candidate is not None:
        # Check if selectors genuinely reduce certificate size
        selectors_help = False
        if baseline_cert is not None and cert_with_selectors is not None:
            # Branch certs should be smaller than baseline for genuine benefit
            selectors_help = (
                cert_with_selectors["size"] < baseline_cert["size"] * 0.8 or
                cert_with_selectors["degree"] < baseline_cert["degree"]
            )
        elif cert_with_selectors is not None and baseline_cert is None:
            # Found cert only with selectors — they definitely help
            selectors_help = True

        if selectors_help:
            return 1  # efficient selectors (e.g. PHP-E LPI)
        return 2  # selectors exist but don't genuinely help (e.g. PHP-C)

    # No auxiliary variables or no verified selectors
    if baseline_cert is not None:
        return 0  # direct cert (e.g. PHP)
    return 3  # hard, no useful selectors (e.g. Tseitin)


# ---------------------------------------------------------------------------
# Discovery Engine
# ---------------------------------------------------------------------------

class DiscoveryEngine:
    """Orchestrates the full selector discovery pipeline.

    Flow:
        1. Baseline: find IPS certificate without selectors
        2. Search: run each strategy to produce candidates
        3. Verify: exhaustive (small n) or algebraic (large n)
        4. Measure: compare cert size with/without selectors
        5. Classify: assign SC level 0/1/2/3
    """

    def __init__(self, config=None):
        self.config = config or DiscoveryConfig()

    def run(self, system_name, axioms, num_vars, var_maps=None):
        """Run full discovery pipeline on a polynomial system.

        Args:
            system_name: Human-readable name like "PHP-E(3)"
            axioms: list of axioms, each a list of (coef, frozenset) terms
            num_vars: total number of boolean variables
            var_maps: dict of variable maps (var_x, var_y, var_s) from axiom builders

        Returns:
            DiscoveryResult with level estimate, candidates, and timings
        """
        if self.config.verbose:
            print(f"=== Discovery: {system_name} ({num_vars} vars, "
                  f"{len(axioms)} axioms) ===")

        result = DiscoveryResult(
            system=system_name, n=num_vars,
            level_estimate=3, candidates_tested=0,
            candidates_verified=0
        )
        timings = {}

        # --- Step 1: Baseline certificate ---
        t0 = time.time()
        baseline = self._find_baseline(axioms, num_vars)
        timings["baseline"] = round(time.time() - t0, 3)

        if baseline is not None:
            result.cert_without_selectors = {
                "degree": baseline["degree"],
                "size": baseline["size"],
            }
            if self.config.verbose:
                print(f"  Baseline cert: degree={baseline['degree']}, "
                      f"size={baseline['size']}")

        # --- Step 2: Run strategies ---
        from selector_complexity.discovery_strategies import get_strategies
        strategies = get_strategies(self.config, axioms, num_vars, var_maps)

        all_candidates = []
        for name, strategy in strategies.items():
            if name not in self.config.strategies:
                continue
            t0 = time.time()
            try:
                candidates = strategy.search()
                all_candidates.extend(candidates)
                if self.config.verbose:
                    print(f"  {name}: {len(candidates)} candidates")
            except Exception as e:
                if self.config.verbose:
                    print(f"  {name}: ERROR - {e}")
            timings[name] = round(time.time() - t0, 3)

        result.candidates_tested = len(all_candidates)

        # --- Step 3: Verify ---
        verifier = SelectorVerifier(num_vars, self.config.max_vars_exhaustive)
        verified = []
        for cand in all_candidates:
            if verifier.verify(cand):
                cand.pre_verified = True
                verified.append(cand)

        result.candidates_verified = len(verified)
        if self.config.verbose:
            print(f"  Verified: {len(verified)}/{len(all_candidates)}")

        # --- Step 4: Measure quality ---
        # For SC hierarchy: only consider selectors using auxiliary variables
        # when the system has them. Generic x-variable selectors work for
        # any boolean system and don't reflect the structure.
        aux_indices = set()
        if var_maps:
            for key in var_maps:
                if key != "var_x":
                    aux_indices.update(var_maps[key].values())

        def uses_auxiliary(cand):
            """Check if any selector monomial involves auxiliary variables."""
            if not aux_indices:
                return True  # no auxiliary vars → all candidates count
            for terms in cand.selectors.values():
                for _, monom in terms:
                    if monom & aux_indices:  # frozenset intersection
                        return True
            return False

        # Prioritize aux-variable candidates, fall back to all
        aux_verified = [c for c in verified if uses_auxiliary(c)]
        search_pool = aux_verified if aux_verified else verified

        best = None
        best_cert = None
        if search_pool:
            measurer = QualityMeasurer(self.config)
            for cand in search_pool:
                cert = measurer.measure(axioms, num_vars, cand, baseline)
                if cert is not None:
                    if best_cert is None or cert["size"] < best_cert["size"]:
                        best = cand
                        best_cert = cert

            # If no cert comparison possible, pick smallest verified candidate
            if best is None:
                best = min(search_pool, key=lambda c: c.total_size)

        # --- Step 5: Classify ---
        level = classify_level(baseline, best, best_cert, self.config,
                               var_maps)
        result.level_estimate = level

        if best is not None:
            result.best_selector = {
                "num_selectors": best.num_selectors,
                "total_size": best.total_size,
                "degree": best.degree,
                "source": best.source,
                "polynomials": _serialize_selectors(best.selectors),
            }
        if best_cert is not None:
            result.cert_with_selectors = {
                "degree": best_cert["degree"],
                "size": best_cert["size"],
            }

        result.timings = timings

        if self.config.verbose:
            print(f"  Level estimate: SC({level})")
            print(f"  Timings: {timings}")

        return result

    def _find_baseline(self, axioms, num_vars):
        """Find IPS certificate without selectors at increasing degree."""
        for d in range(1, self.config.max_degree + 1):
            total_monoms = sum(comb(num_vars, k) for k in range(d + 1))
            if total_monoms > self.config.monomial_cap:
                break

            if num_vars > 40 and self.config.use_aip:
                cert = _solve_large_with_accordion(
                    axioms, num_vars, d, use_aip=True)
            else:
                cert = find_certificate(axioms, num_vars, d)
            if cert is not None:
                return cert
        return None
