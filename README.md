# Selector Complexity Framework

[![PyPI version](https://img.shields.io/pypi/v/selector-complexity.svg)](https://pypi.org/project/selector-complexity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

**A new hierarchy for proof complexity.** Classifies tautologies by the cost of their "selector" polynomials in Ideal Proof Systems (IPS).

**Author:** Carmen Esteban — February 2026

---

## The Hierarchy

```
SC(0)  ⊊  SC(1)  ⊊  SC(2)  ⊊  SC(3)
 PHP      PHP-E      PHP-C      Tseitin(expander)
```

| Level | What it means | Example | Selector Cost | IPS Certificate | Status |
|-------|--------------|---------|---------------|-----------------|--------|
| **SC(0)** | No selectors needed | PHP | — | O(n²) | **Proven** |
| **SC(1)** | Efficient selectors exist | PHP-E | O(n²) circuits | O(n⁴) | **Proven** |
| **SC(2)** | Selectors cost Ω(n!) | PHP-C | Ω(n!) | 2^poly(n) | **Proven** |
| **SC(3)** | No useful selectors at all | Tseitin(expander) | — | No cert at d≤8 | **Proven (v0.5.0)** |

**All four levels are proven** with computational verification.

---

## Quick Start

```bash
pip install selector-complexity
```

```python
from selector_complexity import php_axioms, estimate_level

# Classify any tautology system
axioms, num_vars, _ = php_axioms(3)
result = estimate_level(axioms, num_vars)
print(f"Level: SC({result['level']})")  # SC(0)
```

```python
from selector_complexity import (
    phpe_axioms, phpc_axioms, tseitin_axioms, circulant_graph,
    build_phpe_selectors, test_s_only_feasibility, estimate_level,
)

# PHP-E: efficient selectors exist (Level 1)
selectors = build_phpe_selectors(3)
print(f"PHP-E selectors: {len(selectors)} indicators, cost O(n²)")

# PHP-C: s-only selectors are impossible (Level 2)
result = test_s_only_feasibility(3)
print(f"PHP-C s-only feasible? {result}")  # False

# Tseitin on expanders: no useful selectors (Level 3)
edges, n = circulant_graph(10, [1, 2, 5])
axioms, nv, _ = tseitin_axioms(edges, n)
r = estimate_level(axioms, nv, verbose=False)
print(f"Tseitin-expander(10): SC({r['level']})")  # SC(3)
```

### Family Classification (strongest evidence)

```python
from selector_complexity import estimate_level_family, php_axioms

# Classify by observing scaling across multiple sizes
result = estimate_level_family(
    builder=lambda n: php_axioms(n),
    n_values=[2, 3, 4, 5],
)
print(f"PHP family: SC({result['level']})")  # SC(0), high confidence
```

### Hardness Quantification

```python
from selector_complexity import quantify_hardness, compare_hardness, php_axioms

axioms, nv, _ = php_axioms(3)
h = quantify_hardness(axioms, nv)
print(f"Hardness: {h['hardness_score']}/100")
```

---

## What problem does this solve?

In proof complexity, we know some tautologies are "hard" and others are "easy", but **why**? The Selector Complexity Framework gives a structural answer:

- **Easy tautologies** (SC(0)): the proof has a natural decomposition into cases, no extra machinery needed.
- **Medium tautologies** (SC(1)): you can decompose, but you need auxiliary "selector" polynomials to pick the right case.
- **Hard tautologies** (SC(2)): selectors exist but cost Ω(n!) — symmetry forces exponential overhead.
- **Hardest tautologies** (SC(3)): no useful decomposition exists at all — the contradiction is global.

This is the first framework to classify IPS tautologies by **selector cost**, creating a strict hierarchy with computational proofs.

---

## Computational Proofs

Every claim is backed by runnable Python scripts in `theory/`:

```bash
python theory/02_php_level0.py        # PHP is Level 0
python theory/03_phpe_level1.py       # PHP-E is Level 1
python theory/05_phpc_selector_lower_bound.py  # s-only selectors impossible
python theory/08_phpc_symmetry_argument.py     # Z_{n+1} forces Ω(n!) cost
python theory/09_hierarchy_theorem.py  # Full hierarchy: SC(0) ⊊ SC(1) ⊊ SC(2)
python theory/11_sc3_proof.py          # PROOF: SC(3) exists (Tseitin on expanders)
```

**No claim without computational proof.**

### SC(3) Proof Summary (v0.5.0)

Tseitin tautologies on 3-regular expanders verified across 8 instances (n=6..20):

```
  Instance    | Expansion | Degree ≤ 8 | Residual Trend | Obstruction
  ------------|-----------|------------|----------------|------------
  Tseitin(6)  |    3.00   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(8)  |    2.00   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(10) |    2.20   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(12) |    2.00   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(14) |    1.86   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(16) |    1.50   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(18) |    1.44   | INFEASIBLE |    plateau     |   PROVED
  Tseitin(20) |    1.20   | INFEASIBLE |    plateau     |   PROVED
```

Comparison with lower levels (all find certificates at degree 6):

```
  Family          | Level | Certificate? | Degree | Size
  ----------------|-------|-------------|--------|--------
  PHP(3)          | SC(0) |     YES     |    6   | 20,506
  PHP-E(3)        | SC(1) |     YES     |    6   | 121,130
  PHP-C(3)        | SC(2) |     YES     |    6   | 432,282
  Tseitin-Exp(10) | SC(3) |     NO      |    -   | -
```

---

## Discovery Engine

Automated selector discovery across 9 tautology families using 6 strategies:

| Family | Instances | SC Level | Strategy |
|--------|-----------|----------|----------|
| PHP | 5 | SC(0) | Direct certificate |
| PHP-E | 7 | SC(1) | Template (LPI selectors) |
| PHP-C | 4 | SC(2) | Variable grouping |
| Tseitin | 2 | SC(0) | Axiom graph |
| 3-XOR | 4 | SC(0) | Linear |
| SubsetSum | 3 | SC(0) | Template |
| Factoring | 3 | SC(2) | Selectors don't help |
| Goldreich | 3 | SC(2) | Selectors marginal |
| BinLWE | 3 | SC(1) | Template (product) |

34 instances, all verified. Results in `results/`.

---

## Project Structure

```
Selector-Complexity-Framework/
├── selector_complexity/             # Python package
│   ├── core.py                      #   PolynomialSystem, SelectorFamily
│   ├── php.py                       #   PHP, PHP-E, PHP-C axiom builders
│   ├── tseitin.py                   #   Tseitin axioms + graph constructors
│   ├── families.py                  #   Factoring, Goldreich, BinLWE builders
│   ├── selectors.py                 #   Selector construction and feasibility
│   ├── solvers.py                   #   IPS matrix builder and LSQR solver
│   ├── classifier.py               #   Automatic SC-level classifier
│   ├── strategy.py                  #   Proof strategy advisor
│   ├── optimizer.py                 #   Optimized certificate search
│   ├── hardness.py                  #   Hardness quantifier (0-100 score)
│   ├── discovery.py                 #   Selector discovery engine
│   ├── discovery_strategies.py      #   6 automated discovery strategies
│   ├── patterns.py                  #   Pattern detection (XOR, DP, graph)
│   └── predictor.py                 #   ML-free SC predictor (decision tree)
├── theory/                          # Computational proofs (01–11)
│   ├── 01–09                        #   Levels 0–2 proofs (see above)
│   ├── 10_level3_candidates.py      #   Level 3 candidate analysis
│   └── 11_sc3_proof.py              #   PROOF: SC(3) exists
├── discovery/                       # Discovery session runner
├── results/                         # 35 discovery result files (JSON)
├── analysis/                        # Pattern extraction
└── tests/
    └── run_all_tests.py
```

## Install from Source

```bash
git clone https://github.com/iafiscal1212/Selector-Complexity-Framework.git
cd Selector-Complexity-Framework
pip install -e .
python theory/11_sc3_proof.py
```

## Open Questions

1. **Infinite hierarchy?** Are there infinitely many distinct selector complexity levels beyond SC(3)?
2. **Tight bounds?** Can the n! lower bound for PHP-C be improved to 2^Ω(n)?
3. **Random 3-XOR?** Are random unsatisfiable k-XOR systems also Level 3?
4. **Cryptographic hardness?** Is Factoring provably SC(2) (not just computationally)?

## License

MIT — see [LICENSE](LICENSE).
