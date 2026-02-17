"""
SC Level Predictor
==================

Lightweight predictor that estimates SC level from structural features
without running the expensive IPS certificate search.

Uses a decision tree by thresholds, no external ML dependencies.

Classes:
    SCPredictor           -- decision tree predictor

Functions:
    extract_features      -- 17 structural features from axiom system
    generate_training_data -- generate (features, sc_level) from known families

Author: Carmen Esteban
License: MIT
"""

import math
from itertools import combinations


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_features(axioms, num_vars):
    """Extract 17 structural features from a polynomial axiom system.

    Features are computed without building IPS matrices, making this
    very fast (linear in axiom size).

    Parameters
    ----------
    axioms : list of list of (coef, frozenset) tuples
    num_vars : int

    Returns
    -------
    dict with 17 feature keys:
        num_axioms, num_vars, max_degree, avg_degree, var_density,
        connectivity, expansion, axiom_var_ratio, degree_1_count,
        degree_2_count, degree_higher_count, avg_clause_width,
        max_clause_width, graph_components, max_component_size,
        spectral_gap_estimate, coef_magnitude_avg
    """
    if not axioms:
        return {k: 0 for k in [
            'num_axioms', 'num_vars', 'max_degree', 'avg_degree',
            'var_density', 'connectivity', 'expansion', 'axiom_var_ratio',
            'degree_1_count', 'degree_2_count', 'degree_higher_count',
            'avg_clause_width', 'max_clause_width', 'graph_components',
            'max_component_size', 'spectral_gap_estimate', 'coef_magnitude_avg',
        ]}

    # Degree stats
    degrees = []
    clause_widths = []  # number of variables per axiom
    all_coefs = []

    for ax in axioms:
        deg = max((len(m) for _, m in ax), default=0)
        degrees.append(deg)
        ax_vars = set()
        for c, m in ax:
            ax_vars.update(m)
            all_coefs.append(abs(c))
        clause_widths.append(len(ax_vars))

    max_degree = max(degrees)
    avg_degree = sum(degrees) / len(degrees)
    degree_1_count = sum(1 for d in degrees if d <= 1)
    degree_2_count = sum(1 for d in degrees if d == 2)
    degree_higher_count = sum(1 for d in degrees if d > 2)

    avg_clause_width = sum(clause_widths) / len(clause_widths)
    max_clause_width = max(clause_widths)

    coef_magnitude_avg = sum(all_coefs) / len(all_coefs) if all_coefs else 0.0

    # Variable usage
    vars_used = set()
    for ax in axioms:
        for _, m in ax:
            vars_used.update(m)
    var_density = len(vars_used) / max(num_vars, 1)

    # Constraint graph
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

    # Connectivity
    num_edges = sum(len(nb) for nb in adj.values()) // 2
    max_edges = num_vars * (num_vars - 1) // 2
    connectivity = num_edges / max(max_edges, 1)

    # Expansion: avg degree / (num_vars - 1)
    active_degrees = [len(adj.get(v, set())) for v in range(num_vars)]
    avg_graph_degree = sum(active_degrees) / max(num_vars, 1)
    expansion = avg_graph_degree / max(num_vars - 1, 1)

    axiom_var_ratio = len(axioms) / max(num_vars, 1)

    # Connected components via BFS
    visited = set()
    components = []
    for start in range(num_vars):
        if start in visited:
            continue
        if not adj.get(start):
            visited.add(start)
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
        if component:
            components.append(component)

    graph_components = len(components) if components else 1
    max_component_size = max(len(c) for c in components) if components else 0

    # Spectral gap estimate (only for small systems)
    spectral_gap = 0.0
    if num_vars <= 200 and num_vars > 1:
        try:
            import numpy as np
            # Build adjacency matrix
            A = np.zeros((num_vars, num_vars))
            for v in range(num_vars):
                for nb in adj.get(v, set()):
                    A[v][nb] = 1.0

            deg_vec = A.sum(axis=1)
            max_d = deg_vec.max()
            if max_d > 0:
                # Normalized Laplacian eigenvalues
                D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg_vec, 1)))
                L_norm = np.eye(num_vars) - D_inv_sqrt @ A @ D_inv_sqrt
                eigvals = np.linalg.eigvalsh(L_norm)
                eigvals.sort()
                # Spectral gap = second smallest eigenvalue
                if len(eigvals) >= 2:
                    spectral_gap = float(eigvals[1])
        except Exception:
            spectral_gap = 0.0

    return {
        'num_axioms': len(axioms),
        'num_vars': num_vars,
        'max_degree': max_degree,
        'avg_degree': round(avg_degree, 3),
        'var_density': round(var_density, 3),
        'connectivity': round(connectivity, 4),
        'expansion': round(expansion, 4),
        'axiom_var_ratio': round(axiom_var_ratio, 3),
        'degree_1_count': degree_1_count,
        'degree_2_count': degree_2_count,
        'degree_higher_count': degree_higher_count,
        'avg_clause_width': round(avg_clause_width, 2),
        'max_clause_width': max_clause_width,
        'graph_components': graph_components,
        'max_component_size': max_component_size,
        'spectral_gap_estimate': round(spectral_gap, 4),
        'coef_magnitude_avg': round(coef_magnitude_avg, 4),
    }


# =====================================================================
# TRAINING DATA GENERATION
# =====================================================================

def generate_training_data():
    """Generate (features, sc_level) pairs from the 7 known families.

    Uses small instances for speed. Returns list of (features_dict, int_level).
    Level '3?' is encoded as 3.

    Returns
    -------
    list of (dict, int)
    """
    from selector_complexity.php import php_axioms, phpe_axioms, phpc_axioms
    from selector_complexity.tseitin import tseitin_axioms, circulant_graph

    data = []

    # PHP -> SC(0)
    for n in [2, 3, 4]:
        ax, nv = php_axioms(n)[:2]
        data.append((extract_features(ax, nv), 0))

    # PHP-E -> SC(1)
    for n in [2, 3]:
        ax, nv = phpe_axioms(n)[:2]
        data.append((extract_features(ax, nv), 1))

    # PHP-C -> SC(2)
    for n in [2, 3]:
        ax, nv = phpc_axioms(n)[:2]
        data.append((extract_features(ax, nv), 2))

    # Tseitin-cycle -> SC(0)
    for n in [6, 8, 10]:
        edges, nv_graph = circulant_graph(n, [1])
        ax, nv = tseitin_axioms(edges, nv_graph)[:2]
        data.append((extract_features(ax, nv), 0))

    # Tseitin-expander -> SC(3?)
    for n in [8, 10]:
        edges, nv_graph = circulant_graph(n, [1, 3])
        ax, nv = tseitin_axioms(edges, nv_graph)[:2]
        data.append((extract_features(ax, nv), 3))

    return data


# =====================================================================
# SC PREDICTOR (decision tree by thresholds, no ML dependency)
# =====================================================================

class SCPredictor:
    """Lightweight SC-level predictor using a decision tree by thresholds.

    No external ML dependencies. Learns threshold splits from training data.

    Usage
    -----
    pred = SCPredictor()
    pred.fit_from_landscape()
    result = pred.predict(features)
    """

    # Feature keys used for splitting
    FEATURE_KEYS = [
        'num_axioms', 'num_vars', 'max_degree', 'avg_degree',
        'var_density', 'connectivity', 'expansion', 'axiom_var_ratio',
        'degree_1_count', 'degree_2_count', 'degree_higher_count',
        'avg_clause_width', 'max_clause_width', 'graph_components',
        'max_component_size', 'spectral_gap_estimate', 'coef_magnitude_avg',
    ]

    def __init__(self):
        self._fitted = False
        self._means = {}     # {level: {feature: mean_value}}
        self._splits = []    # list of (feature, threshold, below_level, above_level)
        self._level_counts = {}

    def fit(self, training_data):
        """Fit the predictor from training data.

        Parameters
        ----------
        training_data : list of (dict, int)
            Each element is (features_dict, sc_level).
        """
        if not training_data:
            return

        # Group by level
        by_level = {}
        for features, level in training_data:
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(features)

        self._level_counts = {lv: len(samples) for lv, samples in by_level.items()}

        # Compute means per level per feature
        self._means = {}
        for level, samples in by_level.items():
            self._means[level] = {}
            for key in self.FEATURE_KEYS:
                vals = [s[key] for s in samples if key in s]
                self._means[level][key] = sum(vals) / len(vals) if vals else 0.0

        # Find best splits: for each feature, find threshold that best
        # separates levels
        self._splits = []
        levels = sorted(by_level.keys())

        for key in self.FEATURE_KEYS:
            all_vals = []
            for level, samples in by_level.items():
                for s in samples:
                    if key in s:
                        all_vals.append((s[key], level))
            if not all_vals:
                continue

            all_vals.sort(key=lambda x: x[0])

            # Try each midpoint as threshold
            best_score = -1
            best_split = None
            for i in range(len(all_vals) - 1):
                if all_vals[i][0] == all_vals[i + 1][0]:
                    continue
                threshold = (all_vals[i][0] + all_vals[i + 1][0]) / 2

                below = [lv for v, lv in all_vals if v <= threshold]
                above = [lv for v, lv in all_vals if v > threshold]

                if not below or not above:
                    continue

                # Score: how well does this split separate levels?
                below_mode = max(set(below), key=below.count)
                above_mode = max(set(above), key=above.count)

                if below_mode == above_mode:
                    continue

                accuracy = (
                    sum(1 for lv in below if lv == below_mode) +
                    sum(1 for lv in above if lv == above_mode)
                ) / len(all_vals)

                if accuracy > best_score:
                    best_score = accuracy
                    best_split = (key, threshold, below_mode, above_mode, accuracy)

            if best_split and best_split[4] > 0.6:
                self._splits.append(best_split)

        # Sort splits by accuracy (best first)
        self._splits.sort(key=lambda x: x[4], reverse=True)

        self._fitted = True

    def predict(self, features):
        """Predict SC level from features.

        Parameters
        ----------
        features : dict
            Output of extract_features().

        Returns
        -------
        dict
            'predicted_level' : int
            'confidence' : float (0-1)
            'reasoning' : str
            'feature_signals' : list of str
        """
        if not self._fitted:
            return {
                'predicted_level': None,
                'confidence': 0.0,
                'reasoning': 'Predictor not fitted.',
                'feature_signals': [],
            }

        # Vote from each split rule
        votes = {}
        signals = []

        for key, threshold, below_lv, above_lv, accuracy in self._splits:
            val = features.get(key, 0)
            if val <= threshold:
                vote = below_lv
                direction = '<='
            else:
                vote = above_lv
                direction = '>'

            weight = accuracy
            votes[vote] = votes.get(vote, 0) + weight
            signals.append("{} {:.3f} {} {:.3f} -> SC({})".format(
                key, val, direction, threshold, vote))

        if not votes:
            # Fallback: nearest-mean classifier
            best_level = 0
            best_dist = float('inf')
            for level, means in self._means.items():
                dist = sum((features.get(k, 0) - means.get(k, 0)) ** 2
                           for k in self.FEATURE_KEYS)
                if dist < best_dist:
                    best_dist = dist
                    best_level = level
            return {
                'predicted_level': best_level,
                'confidence': 0.3,
                'reasoning': 'Nearest-mean fallback (no splits matched).',
                'feature_signals': signals,
            }

        # Winner by weighted vote
        total_weight = sum(votes.values())
        predicted = max(votes, key=votes.get)
        confidence = votes[predicted] / total_weight if total_weight > 0 else 0.0

        reasoning = "Predicted SC({}) with {:.0%} vote confidence from {} split rules.".format(
            predicted, confidence, len(self._splits))

        return {
            'predicted_level': predicted,
            'confidence': round(confidence, 3),
            'reasoning': reasoning,
            'feature_signals': signals[:5],  # top 5 signals
        }

    def fit_from_landscape(self):
        """Generate training data from known families and fit in one step."""
        data = generate_training_data()
        self.fit(data)
        return self
