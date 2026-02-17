"""
Selector Complexity Framework
==============================

Classifies tautologies by the complexity of their selector
families in IPS (Ideal Proof Systems).

Levels:
  0 - Polynomial certificates, no selectors needed (e.g. PHP)
  1 - Efficient selectors in auxiliary variables (e.g. PHP-E)
  2+- Selectors require original variables, expensive (e.g. PHP-C)

Author: Carmen Esteban
License: MIT
"""

__version__ = "0.3.0"  # computational optimizations: patterns, incremental, predictor, parallel
__author__ = "Carmen Esteban"

from selector_complexity.core import PolynomialSystem, SelectorFamily
from selector_complexity.php import php_axioms, phpe_axioms, phpc_axioms
from selector_complexity.solvers import (
    build_matrix, find_certificate,
    build_matrix_tuples, find_certificate_blocked,
    IncrementalIPSState, incremental_certificate_search,
)
from selector_complexity.selectors import (
    build_phpe_selectors,
    build_phpc_explicit_selectors,
    enumerate_vc,
    test_s_only_feasibility,
)
from selector_complexity.classifier import estimate_level, estimate_level_family
from selector_complexity.strategy import recommend_strategy
from selector_complexity.optimizer import (
    predict_min_degree, optimized_certificate_search,
    parallel_certificate_search,
)
from selector_complexity.hardness import quantify_hardness, compare_hardness, hardness_report
from selector_complexity.tseitin import (
    tseitin_axioms,
    petersen_graph,
    cube_graph,
    circulant_graph,
    random_regular,
)
from selector_complexity.patterns import (
    detect_xor_structure,
    detect_subset_sum_structure,
    detect_graph_topology,
    detect_patterns,
)
from selector_complexity.predictor import (
    extract_features,
    SCPredictor,
    generate_training_data,
)
