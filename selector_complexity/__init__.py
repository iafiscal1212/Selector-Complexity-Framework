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

__version__ = "0.1.0"
__author__ = "Carmen Esteban"

from selector_complexity.core import PolynomialSystem, SelectorFamily
from selector_complexity.php import php_axioms, phpe_axioms, phpc_axioms
from selector_complexity.solvers import build_matrix, find_certificate
from selector_complexity.selectors import (
    build_phpe_selectors,
    build_phpc_explicit_selectors,
    enumerate_vc,
    test_s_only_feasibility,
)
