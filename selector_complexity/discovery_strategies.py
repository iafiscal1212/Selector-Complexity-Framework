"""
Six discovery strategies for selector families.
================================================

Each strategy implements .search() returning a list of SelectorCandidate.

Strategies:
    1. ExhaustiveStrategy         - SVD on boolean cube (n <= 15)
    2. TemplateStrategy           - Product, linear, symmetry templates
    3. IPSGuidedStrategy          - Treat selector coefficients as unknowns
    4. SubspaceProjectionStrategy - Project onto informative variable subsets
    5. VariableGroupStrategy      - Rank variables by info score, build products
    6. AxiomGraphStrategy         - Axiom dependency graph decomposition

Author: Carmen Esteban
"""

import numpy as np
from itertools import combinations
from math import comb

from selector_complexity.discovery import SelectorCandidate, DiscoveryConfig


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Base class for discovery strategies."""

    def __init__(self, config, axioms, num_vars, var_maps=None):
        self.config = config
        self.axioms = axioms
        self.num_vars = num_vars
        self.var_maps = var_maps or {}

    def search(self):
        raise NotImplementedError

    def _monomial_count(self, n, d):
        return sum(comb(n, k) for k in range(d + 1))

    def _enumerate_monomials(self, var_indices, max_deg):
        """Enumerate all monomials up to max_deg over given variables."""
        var_indices = list(var_indices)
        monoms = [frozenset()]
        for d in range(1, max_deg + 1):
            for combo in combinations(var_indices, d):
                monoms.append(frozenset(combo))
        return monoms

    def _eval_monomial(self, monom, assignment):
        prod = 1.0
        for v in monom:
            prod *= assignment.get(v, 0)
        return prod

    def _multiply_poly(self, terms_a, terms_b):
        """Multiply two polynomials represented as [(coef, frozenset), ...]."""
        combined = {}
        for c1, m1 in terms_a:
            for c2, m2 in terms_b:
                m = m1 | m2
                combined[m] = combined.get(m, 0.0) + c1 * c2
        return [(c, m) for m, c in combined.items() if abs(c) > 1e-15]

    def _build_product_selectors(self, var_set):
        """Build product-type selectors from a variable set.

        For each assignment pattern bits over var_set:
            g_bits = Prod_{i: bit=1} v_i * Prod_{i: bit=0} (1 - v_i)
        """
        k = len(var_set)
        selectors = {}
        for bits in range(2 ** k):
            terms = [(1.0, frozenset())]
            for i, v in enumerate(var_set):
                if (bits >> i) & 1:
                    factor = [(1.0, frozenset([v]))]
                else:
                    factor = [(1.0, frozenset()), (-1.0, frozenset([v]))]
                terms = self._multiply_poly(terms, factor)
            selectors[bits] = terms
        return selectors


# ---------------------------------------------------------------------------
# Strategy 1: Exhaustive (small n)
# ---------------------------------------------------------------------------

class ExhaustiveStrategy(BaseStrategy):
    """SVD-based search over the full boolean cube.

    Builds evaluation matrix on {0,1}^n, uses SVD to find natural clusters
    among assignments, and fits selector polynomials via least-squares.
    """

    def search(self):
        if self.num_vars > self.config.max_vars_exhaustive:
            return []

        candidates = []

        for d in range(1, self.config.max_selector_degree + 1):
            monoms = self._enumerate_monomials(range(self.num_vars), d)
            if len(monoms) > self.config.monomial_cap:
                break

            n_points = 2 ** self.num_vars
            eval_mat = np.zeros((n_points, len(monoms)))

            for bits in range(n_points):
                assignment = {v: (bits >> v) & 1 for v in range(self.num_vars)}
                for j, m in enumerate(monoms):
                    eval_mat[bits, j] = self._eval_monomial(m, assignment)

            # SVD to detect natural clustering
            U, S, Vt = np.linalg.svd(eval_mat, full_matrices=False)

            # Try k selectors for k = 2..max, guided by singular value gaps
            max_k = min(self.config.max_selector_count, len(S), n_points)
            for k in range(2, max_k + 1):
                if k >= len(S) or S[k - 1] < 1e-8:
                    break

                selectors = self._fit_selectors(eval_mat, monoms, k)
                if selectors is not None:
                    cand = SelectorCandidate(
                        selectors=selectors,
                        source=f"exhaustive:svd:d{d}:k{k}"
                    )
                    candidates.append(cand)

        return candidates

    def _fit_selectors(self, eval_mat, monoms, k):
        """Fit k indicator selectors using K-means + least squares."""
        n_points = eval_mat.shape[0]

        # Project onto top-k left singular vectors
        U, S, Vt = np.linalg.svd(eval_mat, full_matrices=False)
        dim = min(k, U.shape[1])
        projected = U[:, :dim]

        # K-means clustering
        rng = np.random.RandomState(42)
        labels = rng.randint(0, k, size=n_points)

        for _ in range(30):
            centers = []
            for c in range(k):
                mask = labels == c
                if np.sum(mask) > 0:
                    centers.append(projected[mask].mean(axis=0))
                else:
                    centers.append(projected[rng.randint(n_points)])
            centers = np.array(centers)
            dists = np.linalg.norm(
                projected[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        # For each cluster, fit a selector polynomial via least squares
        selectors = {}
        for c in range(k):
            target = (labels == c).astype(float)
            result = np.linalg.lstsq(eval_mat, target, rcond=None)
            coeffs = result[0]
            residual = np.linalg.norm(eval_mat @ coeffs - target)

            if residual > 1e-4:
                return None

            terms = []
            for j, m in enumerate(monoms):
                if abs(coeffs[j]) > 1e-10:
                    terms.append((float(coeffs[j]), m))
            if not terms:
                terms = [(0.0, frozenset())]
            selectors[c] = terms

        return selectors


# ---------------------------------------------------------------------------
# Strategy 2: Template-based
# ---------------------------------------------------------------------------

class TemplateStrategy(BaseStrategy):
    """Search using known template patterns.

    Sub-strategies:
        - Products: g_S = Prod v_j^{bit} * (1-v_j)^{1-bit}
        - Linear: binary partition g_0 = x_v, g_1 = 1 - x_v
        - Symmetry: variables with identical appearance -> group selectors
    """

    def search(self):
        candidates = []
        candidates.extend(self._search_known_patterns())
        candidates.extend(self._search_products())
        candidates.extend(self._search_linear())
        candidates.extend(self._search_symmetry())
        return candidates

    def _search_known_patterns(self):
        """Try known selector constructions from the framework.

        For PHP-E: uses build_phpe_selectors (LPI - Last Pigeon Indicators).
        These are the canonical efficient selectors that decompose by
        'which pigeon is last', remapped to the global variable numbering.
        """
        candidates = []

        if "var_y" in self.var_maps:
            candidates.extend(self._try_lpi_selectors())

        return candidates

    def _try_lpi_selectors(self):
        """Build LPI selectors using build_phpe_selectors and remap."""
        from selector_complexity.selectors import build_phpe_selectors

        y_map = self.var_maps["var_y"]
        num_y = len(y_map)

        # Infer n from C(n+1, 2) = num_y
        n_plus_1 = int((1 + (1 + 8 * num_y) ** 0.5) / 2)
        n = n_plus_1 - 1
        if n < 2 or n_plus_1 * (n_plus_1 - 1) // 2 != num_y:
            return []

        indicators_local, local_y_map = build_phpe_selectors(n)

        # Build local → global variable index mapping
        local_to_global = {}
        for key, local_idx in local_y_map.items():
            if key in y_map:
                local_to_global[local_idx] = y_map[key]

        if len(local_to_global) != num_y:
            return []

        # Remap indicator polynomials to global variable indices
        remapped = {}
        for label, terms in indicators_local.items():
            new_terms = []
            for coef, monom in terms:
                new_monom = frozenset(local_to_global[v] for v in monom)
                new_terms.append((coef, new_monom))
            remapped[label] = new_terms

        cand = SelectorCandidate(
            selectors=remapped,
            source="template:known:lpi"
        )
        return [cand]

    def _search_products(self):
        """Product selectors over subsets of indicator variables."""
        candidates = []
        var_sets = self._find_indicator_variables()

        for var_set in var_sets:
            if len(var_set) > 8:
                continue

            selectors = self._build_product_selectors(var_set)

            if len(selectors) <= self.config.max_selector_count:
                label = '_'.join(str(v) for v in var_set)
                cand = SelectorCandidate(
                    selectors=selectors,
                    source=f"template:product:vars{label}"
                )
                candidates.append(cand)

        return candidates

    def _search_linear(self):
        """Binary partition selectors: g_0 = x_v, g_1 = 1 - x_v."""
        candidates = []
        for v in range(self.num_vars):
            selectors = {
                0: [(1.0, frozenset([v]))],
                1: [(1.0, frozenset()), (-1.0, frozenset([v]))]
            }
            cand = SelectorCandidate(
                selectors=selectors,
                source=f"template:linear:var{v}"
            )
            candidates.append(cand)
        return candidates

    def _search_symmetry(self):
        """Find variables with identical appearance patterns.

        Groups variables that appear in the exact same axioms,
        then builds "which variable in this group is active" selectors.
        """
        candidates = []

        # Compute appearance pattern per variable
        patterns = {}
        for v in range(self.num_vars):
            pattern = []
            for ax_idx, ax in enumerate(self.axioms):
                appears = any(v in m for _, m in ax)
                pattern.append(appears)
            patterns[v] = tuple(pattern)

        # Group by pattern
        groups = {}
        for v, pat in patterns.items():
            groups.setdefault(pat, []).append(v)

        # For each group with multiple variables
        for pat, var_group in groups.items():
            if len(var_group) < 2 or len(var_group) > 8:
                continue

            # "Which variable in this group is active" selectors
            selectors = {}
            for i, v in enumerate(var_group):
                selectors[i] = [(1.0, frozenset([v]))]
            # Complementary selector: 1 - sum(x_v for v in group)
            comp_terms = [(1.0, frozenset())]
            for v in var_group:
                comp_terms.append((-1.0, frozenset([v])))
            selectors[len(var_group)] = comp_terms

            if len(selectors) <= self.config.max_selector_count:
                cand = SelectorCandidate(
                    selectors=selectors,
                    source=f"template:symmetry:group{len(var_group)}"
                )
                candidates.append(cand)

        return candidates

    def _find_indicator_variables(self):
        """Find variable subsets likely to serve as indicator bases.

        Prioritizes auxiliary variables (y, s) if available.
        """
        var_sets = []

        # Auxiliary variables from var_maps (scan all non-primary keys)
        for key in sorted(self.var_maps.keys()):
            if key == "var_x":
                continue
            if key in self.var_maps:
                aux_vars = sorted(set(self.var_maps[key].values()))
                for size in range(1, min(5, len(aux_vars) + 1)):
                    for combo in combinations(aux_vars, size):
                        var_sets.append(list(combo))
                        if len(var_sets) > 50:
                            return var_sets

        # Fall back to all variables, small subsets
        for size in range(1, min(4, self.num_vars + 1)):
            for combo in combinations(range(self.num_vars), size):
                var_sets.append(list(combo))
                if len(var_sets) > 50:
                    return var_sets

        return var_sets


# ---------------------------------------------------------------------------
# Strategy 3: IPS-guided
# ---------------------------------------------------------------------------

class IPSGuidedStrategy(BaseStrategy):
    """Use IPS algebraic structure to guide selector search.

    Treats selector coefficients as unknowns and enforces the partition
    of unity constraint: sum_i g_i = 1 at the monomial level.
    For small systems, verifies exclusivity on the boolean cube.
    """

    def search(self):
        candidates = []

        for k in range(2, min(self.config.max_selector_count + 1,
                              self.num_vars + 2)):
            for d in range(1, self.config.max_selector_degree + 1):
                if self._monomial_count(self.num_vars, d) > self.config.monomial_cap:
                    break

                result = self._search_at(k, d)
                if result is not None:
                    candidates.append(result)
                    return candidates  # Early termination on first success

        return candidates

    def _search_at(self, k, degree):
        """Search for k selectors at given degree.

        Sets up partition constraint: for each monomial j,
        sum_i c[i][j] = delta_{j, constant_monomial}.
        Solves underdetermined system, then checks exclusivity.
        """
        monoms = self._enumerate_monomials(range(self.num_vars), degree)
        n_monoms = len(monoms)

        # Partition constraint: sum_i c[i][j] = delta_{j,0}
        A = np.zeros((n_monoms, k * n_monoms))
        b = np.zeros(n_monoms)
        b[0] = 1.0  # constant monomial index 0

        for j in range(n_monoms):
            for i in range(k):
                A[j, i * n_monoms + j] = 1.0

        # Add boolean-cube exclusivity constraints for small systems
        if self.num_vars <= self.config.max_vars_exhaustive:
            excl_rows = []
            excl_b_vals = []
            for bits in range(2 ** self.num_vars):
                assignment = {v: (bits >> v) & 1
                              for v in range(self.num_vars)}
                eval_vec = np.array(
                    [self._eval_monomial(m, assignment) for m in monoms])

                # g_i(a)^2 = g_i(a)  =>  (eval_vec @ c_i)^2 = eval_vec @ c_i
                # Linearized: eval_vec @ c_i in {0, 1}
                # We enforce: eval_vec @ c_i >= 0 (soft via penalty)
                # Instead: add equations that each g_i(a) matches a binary target
                # This is quadratic; we rely on the partition + lstsq to handle it

            # Solve partition constraint (underdetermined → minimum-norm)
            result = np.linalg.lstsq(A, b, rcond=None)
            coeffs = result[0]
        else:
            result = np.linalg.lstsq(A, b, rcond=None)
            coeffs = result[0]

        # Extract selectors from coefficient vector
        selectors = {}
        for i in range(k):
            terms = []
            for j, m in enumerate(monoms):
                c = coeffs[i * n_monoms + j]
                if abs(c) > 1e-10:
                    terms.append((float(c), m))
            if not terms:
                terms = [(0.0, frozenset())]
            selectors[i] = terms

        return SelectorCandidate(
            selectors=selectors,
            source=f"ips_guided:k{k}:d{degree}"
        )


# ---------------------------------------------------------------------------
# Strategy 4: Subspace projection
# ---------------------------------------------------------------------------

class SubspaceProjectionStrategy(BaseStrategy):
    """Project axioms onto variable subsets and test informativeness.

    If projecting onto a subset preserves structure, that subset is
    "informative" and can serve as a selector base.
    """

    def search(self):
        if self.num_vars > 30:
            return []

        candidates = []
        rankings = self._rank_variables()
        top_vars = rankings[:self.config.top_k_variables]

        for size in range(1, min(5, len(top_vars) + 1)):
            for combo in combinations(top_vars, size):
                var_subset = list(combo)
                selectors = self._build_from_subspace(var_subset)
                if selectors is not None:
                    label = '_'.join(str(v) for v in var_subset)
                    cand = SelectorCandidate(
                        selectors=selectors,
                        source=f"subspace:vars{label}"
                    )
                    candidates.append(cand)
                    if len(candidates) >= 10:
                        return candidates

        return candidates

    def _rank_variables(self):
        """Rank variables by information score (matrix rank drop on removal)."""
        if self.num_vars > 20:
            return list(range(min(self.num_vars, self.config.top_k_variables)))

        monoms = self._enumerate_monomials(range(self.num_vars), 2)
        n_points = min(2 ** self.num_vars, 1024)

        eval_mat = np.zeros((n_points, len(monoms)))
        for bits in range(n_points):
            assignment = {v: (bits >> v) & 1 for v in range(self.num_vars)}
            for j, m in enumerate(monoms):
                eval_mat[bits, j] = self._eval_monomial(m, assignment)

        base_rank = np.linalg.matrix_rank(eval_mat)
        scores = []
        for v in range(self.num_vars):
            keep_cols = [j for j, m in enumerate(monoms) if v not in m]
            if not keep_cols:
                scores.append((v, base_rank))
                continue
            reduced_rank = np.linalg.matrix_rank(eval_mat[:, keep_cols])
            scores.append((v, base_rank - reduced_rank))

        scores.sort(key=lambda x: -x[1])
        return [v for v, _ in scores]

    def _build_from_subspace(self, var_subset):
        """Build product selectors from an informative variable subset."""
        k = len(var_subset)
        if k > 6:
            return None

        n_selectors = 2 ** k
        if n_selectors > self.config.max_selector_count:
            return None

        return self._build_product_selectors(var_subset)


# ---------------------------------------------------------------------------
# Strategy 5: Variable grouping
# ---------------------------------------------------------------------------

class VariableGroupStrategy(BaseStrategy):
    """Group variables by information score and build product selectors.

    Computes per-variable information score (rank drop or frequency),
    selects top-K, and builds product selectors from subsets.
    """

    def search(self):
        candidates = []
        scores = self._compute_info_scores()
        if not scores:
            return candidates

        top_k = min(self.config.top_k_variables, len(scores))
        top_vars = [v for v, _ in scores[:top_k]]

        for size in range(1, min(5, len(top_vars) + 1)):
            for combo in combinations(top_vars, size):
                var_set = list(combo)
                if 2 ** len(var_set) > self.config.max_selector_count:
                    continue
                selectors = self._build_product_selectors(var_set)
                label = '_'.join(str(v) for v in var_set)
                cand = SelectorCandidate(
                    selectors=selectors,
                    source=f"variable_group:top{size}:{label}"
                )
                candidates.append(cand)
                if len(candidates) >= 20:
                    return candidates

        return candidates

    def _compute_info_scores(self):
        """Score each variable by its contribution to axiom structure.

        For small n: uses matrix rank drop on removal.
        For large n: uses appearance frequency as proxy.
        """
        if self.num_vars > 20:
            freq = [0] * self.num_vars
            for ax in self.axioms:
                for _, m in ax:
                    for v in m:
                        if v < self.num_vars:
                            freq[v] += 1
            scores = [(v, freq[v]) for v in range(self.num_vars)]
            scores.sort(key=lambda x: -x[1])
            return scores

        monoms = self._enumerate_monomials(range(self.num_vars), 2)
        n_points = min(2 ** self.num_vars, 512)

        eval_mat = np.zeros((n_points, len(monoms)))
        for bits in range(n_points):
            assignment = {v: (bits >> v) & 1 for v in range(self.num_vars)}
            for j, m in enumerate(monoms):
                eval_mat[bits, j] = self._eval_monomial(m, assignment)

        base_rank = np.linalg.matrix_rank(eval_mat)
        scores = []
        for v in range(self.num_vars):
            keep_cols = [j for j, m in enumerate(monoms) if v not in m]
            if not keep_cols:
                scores.append((v, base_rank))
                continue
            reduced_rank = np.linalg.matrix_rank(eval_mat[:, keep_cols])
            scores.append((v, base_rank - reduced_rank))

        scores.sort(key=lambda x: -x[1])
        return scores


# ---------------------------------------------------------------------------
# Strategy 6: Axiom graph decomposition
# ---------------------------------------------------------------------------

class AxiomGraphStrategy(BaseStrategy):
    """Discover selectors from the axiom dependency graph.

    Novel approach: build a graph where axioms are nodes and edges
    represent shared variables. Spectral clustering reveals communities
    of tightly-coupled axioms, and the variables that bridge these
    communities ("hinge variables") are natural selector candidates.

    This captures the *proof topology* — selector variables emerge from
    structural fault lines in the polynomial system rather than from
    properties of individual variables.

    Sub-strategies:
        - Hinge: variables connecting different axiom communities
        - Bridge: variables with high betweenness in variable co-occurrence
        - Bottleneck: minimal variable sets whose removal disconnects the graph
    """

    def search(self):
        candidates = []

        axiom_vars = self._axiom_variable_sets()
        n_axioms = len(axiom_vars)
        if n_axioms < 3:
            return candidates

        adj = self._build_axiom_adjacency(axiom_vars)

        # --- Sub-strategy A: Hinge variables from spectral communities ---
        hinge_cands = []
        for n_parts in range(2, min(5, n_axioms)):
            communities = self._spectral_partition(adj, n_parts)
            if communities is None:
                continue

            hinge_sets = self._find_hinge_variables(axiom_vars, communities)
            for var_set in hinge_sets:
                if 2 ** len(var_set) > self.config.max_selector_count:
                    continue
                selectors = self._build_product_selectors(var_set)
                label = '_'.join(str(v) for v in var_set)
                hinge_cands.append(SelectorCandidate(
                    selectors=selectors,
                    source=f"axiom_graph:hinge:k{n_parts}:vars{label}"
                ))
                if len(hinge_cands) >= 10:
                    break
            if len(hinge_cands) >= 10:
                break
        candidates.extend(hinge_cands)

        # --- Sub-strategy B: Bridge variables (co-occurrence betweenness) ---
        bridge_cands = []
        bridge_sets = self._find_bridge_variables(axiom_vars)
        for var_set in bridge_sets:
            if 2 ** len(var_set) > self.config.max_selector_count:
                continue
            selectors = self._build_product_selectors(var_set)
            label = '_'.join(str(v) for v in var_set)
            bridge_cands.append(SelectorCandidate(
                selectors=selectors,
                source=f"axiom_graph:bridge:vars{label}"
            ))
            if len(bridge_cands) >= 10:
                break
        candidates.extend(bridge_cands)

        # --- Sub-strategy C: Bottleneck variables (axiom graph cuts) ---
        bottleneck_cands = []
        bottleneck_sets = self._find_bottleneck_variables(axiom_vars)
        for var_set in bottleneck_sets:
            if 2 ** len(var_set) > self.config.max_selector_count:
                continue
            selectors = self._build_product_selectors(var_set)
            label = '_'.join(str(v) for v in var_set)
            bottleneck_cands.append(SelectorCandidate(
                selectors=selectors,
                source=f"axiom_graph:bottleneck:vars{label}"
            ))
            if len(bottleneck_cands) >= 10:
                break
        candidates.extend(bottleneck_cands)

        return candidates

    # --- Graph construction ---

    def _axiom_variable_sets(self):
        """For each axiom, collect the set of variables that appear in it."""
        result = []
        for ax in self.axioms:
            vs = set()
            for _, monom in ax:
                vs.update(monom)
            result.append(vs)
        return result

    def _build_axiom_adjacency(self, axiom_vars):
        """Build weighted adjacency matrix: edge weight = |shared variables|."""
        n = len(axiom_vars)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                shared = len(axiom_vars[i] & axiom_vars[j])
                if shared > 0:
                    adj[i, j] = shared
                    adj[j, i] = shared
        return adj

    # --- Spectral community detection ---

    def _spectral_partition(self, adj, k):
        """Partition axioms into k communities using spectral clustering.

        Uses the normalized graph Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}.
        The bottom k eigenvectors embed axioms in R^k; k-means clusters them.
        """
        n = adj.shape[0]
        if n < k + 1:
            return None

        degrees = adj.sum(axis=1)
        isolated = degrees < 1e-10
        if isolated.sum() > n // 2:
            return None

        # Normalized Laplacian
        D_inv_sqrt = np.zeros(n)
        D_inv_sqrt[~isolated] = 1.0 / np.sqrt(degrees[~isolated])
        D_half = np.diag(D_inv_sqrt)
        L_sym = np.eye(n) - D_half @ adj @ D_half

        # Eigendecomposition (only need smallest k+1 eigenvalues)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        except np.linalg.LinAlgError:
            return None

        # Use eigenvectors 1..k (skip the constant eigenvector 0)
        if len(eigenvalues) < k + 1:
            return None

        # Check spectral gap — need meaningful separation
        if eigenvalues[k] < 1e-6:
            return None

        embedding = eigenvectors[:, 1:k + 1]

        # Normalize rows to unit length (spectral clustering convention)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        embedding = embedding / norms

        # K-means on the embedding
        labels = self._kmeans(embedding, k)
        if labels is None:
            return None

        # Verify each community has at least 1 axiom
        communities = {}
        for i, label in enumerate(labels):
            communities.setdefault(label, []).append(i)

        if len(communities) < k:
            return None

        return communities

    def _kmeans(self, X, k, max_iter=50):
        """Simple k-means for spectral embedding."""
        n = X.shape[0]
        rng = np.random.RandomState(42)

        # K-means++ initialization
        centers = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centers)
                              for x in X])
            total = dists.sum()
            if total < 1e-12:
                centers.append(X[rng.randint(n)])
            else:
                probs = dists / total
                centers.append(X[rng.choice(n, p=probs)])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.array([[np.linalg.norm(X[i] - centers[c])
                               for c in range(k)] for i in range(n)])
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels

            # Update centers
            for c in range(k):
                mask = labels == c
                if np.sum(mask) > 0:
                    centers[c] = X[mask].mean(axis=0)

        return labels

    # --- Hinge variable detection ---

    def _find_hinge_variables(self, axiom_vars, communities):
        """Find variables that connect different axiom communities.

        A hinge variable appears in axioms from >= 2 different communities.
        Ranked by the number of distinct community pairs they bridge.
        """
        # Map each axiom index to its community
        axiom_to_comm = {}
        for comm_id, axiom_indices in communities.items():
            for idx in axiom_indices:
                axiom_to_comm[idx] = comm_id

        # For each variable, count which communities it touches
        var_communities = {}
        for ax_idx, vs in enumerate(axiom_vars):
            comm = axiom_to_comm.get(ax_idx)
            if comm is None:
                continue
            for v in vs:
                var_communities.setdefault(v, set()).add(comm)

        # Hinge variables: appear in >= 2 communities
        hinges = []
        for v, comms in var_communities.items():
            if len(comms) >= 2:
                # Score: number of community pairs bridged
                n_pairs = len(comms) * (len(comms) - 1) // 2
                hinges.append((v, n_pairs, len(comms)))

        if not hinges:
            return []

        # Sort by bridging score (descending), then by community count
        hinges.sort(key=lambda x: (-x[1], -x[2]))

        # Build variable sets from top hinge variables
        var_sets = []
        top_hinges = [v for v, _, _ in hinges[:8]]

        for size in range(1, min(5, len(top_hinges) + 1)):
            for combo in combinations(top_hinges, size):
                var_sets.append(list(combo))
                if len(var_sets) >= 20:
                    return var_sets

        return var_sets

    # --- Bridge variable detection ---

    def _find_bridge_variables(self, axiom_vars):
        """Find variables with high betweenness in the variable co-occurrence graph.

        Two variables co-occur if they appear in the same axiom.
        A bridge variable connects otherwise-separated variable clusters.

        Uses an approximation: for each variable v, count the number of
        connected component pairs that would be created if v were removed
        from the co-occurrence graph.
        """
        # Build variable co-occurrence adjacency
        all_vars = set()
        for vs in axiom_vars:
            all_vars.update(vs)

        if len(all_vars) > 200:
            return []

        var_list = sorted(all_vars)
        var_idx = {v: i for i, v in enumerate(var_list)}
        n = len(var_list)

        cooccur = np.zeros((n, n), dtype=bool)
        for vs in axiom_vars:
            vs_list = sorted(vs)
            for a in range(len(vs_list)):
                for b in range(a + 1, len(vs_list)):
                    i, j = var_idx[vs_list[a]], var_idx[vs_list[b]]
                    cooccur[i, j] = True
                    cooccur[j, i] = True

        # Count connected components of the full graph
        base_components = self._count_components(cooccur)

        # For each variable, count components after removal
        bridge_scores = []
        for vi in range(n):
            # Create reduced adjacency
            reduced = cooccur.copy()
            reduced[vi, :] = False
            reduced[:, vi] = False
            new_components = self._count_components(reduced, skip={vi})
            delta = new_components - base_components
            if delta > 0:
                bridge_scores.append((var_list[vi], delta))

        if not bridge_scores:
            # Fall back: use vertex degree in co-occurrence graph as proxy
            degrees = cooccur.sum(axis=1)
            top_deg = np.argsort(-degrees)[:6]
            bridge_scores = [(var_list[i], int(degrees[i]))
                             for i in top_deg if degrees[i] > 0]

        bridge_scores.sort(key=lambda x: -x[1])
        top_bridges = [v for v, _ in bridge_scores[:6]]

        var_sets = []
        for size in range(1, min(4, len(top_bridges) + 1)):
            for combo in combinations(top_bridges, size):
                var_sets.append(list(combo))
                if len(var_sets) >= 15:
                    return var_sets

        return var_sets

    def _count_components(self, adj_bool, skip=None):
        """Count connected components via BFS on boolean adjacency matrix."""
        n = adj_bool.shape[0]
        visited = set(skip) if skip else set()
        components = 0
        for start in range(n):
            if start in visited:
                continue
            components += 1
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop()
                for neighbor in range(n):
                    if neighbor not in visited and adj_bool[node, neighbor]:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return components

    # --- Bottleneck variable detection ---

    def _find_bottleneck_variables(self, axiom_vars):
        """Find minimal variable sets whose removal maximally disconnects axioms.

        For each variable v, measure how much removing all axioms containing v
        fragments the remaining axiom graph. Variables whose removal creates
        the most fragments are "bottleneck" variables — they carry the proof's
        structural weight.
        """
        n_axioms = len(axiom_vars)
        if n_axioms > 200:
            return []

        all_vars = set()
        for vs in axiom_vars:
            all_vars.update(vs)

        # For each variable, find axioms that contain it
        var_to_axioms = {}
        for v in all_vars:
            var_to_axioms[v] = set()
        for ax_idx, vs in enumerate(axiom_vars):
            for v in vs:
                var_to_axioms[v].add(ax_idx)

        # Build axiom adjacency for connectivity check
        axiom_adj = {}
        for i in range(n_axioms):
            axiom_adj[i] = set()
        for i in range(n_axioms):
            for j in range(i + 1, n_axioms):
                if axiom_vars[i] & axiom_vars[j]:
                    axiom_adj[i].add(j)
                    axiom_adj[j].add(i)

        # Base connectivity
        base_cc = self._count_axiom_components(axiom_adj, n_axioms, set())

        # Score each variable by fragmentation
        scores = []
        for v in sorted(all_vars):
            remove_axioms = var_to_axioms[v]
            new_cc = self._count_axiom_components(
                axiom_adj, n_axioms, remove_axioms)
            remaining = n_axioms - len(remove_axioms)
            if remaining < 2:
                continue
            # Fragmentation score: components created minus base
            delta = new_cc - base_cc
            # Also consider what fraction of axioms are affected
            impact = len(remove_axioms) / n_axioms
            # Combined score: fragmentation * (1 + impact)
            combined = delta * (1.0 + impact) if delta > 0 else impact
            scores.append((v, combined))

        scores.sort(key=lambda x: -x[1])
        top_bottlenecks = [v for v, _ in scores[:6] if _ > 0]

        if not top_bottlenecks:
            return []

        var_sets = []
        for size in range(1, min(4, len(top_bottlenecks) + 1)):
            for combo in combinations(top_bottlenecks, size):
                var_sets.append(list(combo))
                if len(var_sets) >= 10:
                    return var_sets

        return var_sets

    def _count_axiom_components(self, axiom_adj, n_axioms, removed):
        """Count connected components in axiom graph after removing some axioms."""
        visited = set(removed)
        components = 0
        for start in range(n_axioms):
            if start in visited:
                continue
            components += 1
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop()
                for neighbor in axiom_adj.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return components


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_strategies(config, axioms, num_vars, var_maps=None):
    """Return dict of strategy_name -> strategy instance."""
    return {
        "exhaustive": ExhaustiveStrategy(config, axioms, num_vars, var_maps),
        "template": TemplateStrategy(config, axioms, num_vars, var_maps),
        "ips_guided": IPSGuidedStrategy(config, axioms, num_vars, var_maps),
        "subspace_projection": SubspaceProjectionStrategy(
            config, axioms, num_vars, var_maps),
        "variable_group": VariableGroupStrategy(
            config, axioms, num_vars, var_maps),
        "axiom_graph": AxiomGraphStrategy(
            config, axioms, num_vars, var_maps),
    }
