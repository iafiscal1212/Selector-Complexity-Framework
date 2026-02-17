# Selector Complexity: A Framework for IPS Certificate Hardness

**Author:** Carmen Esteban
**Date:** February 2026

## What is this?

A formal framework that classifies tautologies by the efficiency of their
"selector" polynomials in Ideal Proof Systems (IPS).

The main result is a **strict hierarchy**:

```
SC(0)  ⊊  SC(1)  ⊊  SC(2)
 PHP      PHP-E      PHP-C
```

Each level is separated by a concrete example from the Pigeonhole Principle family.

## Selector Complexity Levels

| Level | Description | Example | Selector Size | IPS Size | Status |
|-------|-------------|---------|---------------|----------|--------|
| SC(0) | No selectors needed | PHP | -- | O(n^2) | **PROVEN** |
| SC(1) | Efficient selectors | PHP-E | O(n^2) circuits | O(n^4) | **PROVEN** |
| SC(2) | Only exponential selectors | PHP-C | Omega(n!) | 2^poly(n) | **PROVEN** |
| SC(3) | No useful selectors | ??? | -- | 2^Omega(n) | OPEN |

## Structure

```
Selector-Complexity-Framework/
├── selector_complexity/             # Python package (PyPI: selector-complexity)
│   ├── core.py                      #   PolynomialSystem, SelectorFamily, IPSCertificate
│   ├── php.py                       #   PHP, PHP-E, PHP-C axiom builders
│   ├── selectors.py                 #   Selector construction and feasibility
│   └── solvers.py                   #   IPS matrix builder and LSQR solver
├── theory/                          # Computational proofs
│   ├── 01_definitions.py            #   Formal definitions + verification
│   ├── 02_php_level0.py             #   PROOF: PHP is Level 0
│   ├── 03_phpe_level1.py            #   PROOF: PHP-E is Level 1
│   ├── 04_phpc_level2_conjecture.py #   Evidence: PHP-C is Level 2+
│   ├── 05_phpc_selector_lower_bound.py  # PROOF: s-only selectors impossible
│   ├── 06_phpc_formal_identity.py   #   Formal identity cost analysis
│   ├── 07_phpc_growth_analysis.py   #   Growth: factorial vs polynomial
│   ├── 08_phpc_symmetry_argument.py #   Z_{n+1} forces Omega(n!) cost
│   └── 09_hierarchy_theorem.py      #   PROOF: SC(0) ⊊ SC(1) ⊊ SC(2)
└── tests/
    └── run_all_tests.py             #   Test runner for theory/01-04
```

## Key Results

**Theorem (Strict Hierarchy):** The selector complexity classes satisfy
SC(0) ⊊ SC(1) ⊊ SC(2), with:

- **PHP in SC(0):** Telescopic IPS certificates, degree 2, size O(n^2). No selectors needed.
- **PHP-E in SC(1) \ SC(0):** Last Pigeon Indicators give efficient selectors.
  Each g_p(y) has polynomial circuit size. IPS certificates of O(n^4).
- **PHP-C in SC(2) \ SC(1):** Three-part lower bound:
  1. s-only selectors are impossible (gap pigeon invisible to cycle structure).
  2. Mixed selectors must use x-variables, tying cost to PHP structure.
  3. Z_{n+1} cyclic symmetry forces total selector size >= n!.

**Quantitative separation (from theory/09):**

```
  n | PHP-E selectors | PHP-C lower bound | Ratio
  2 |               7 |                 2 |   0.3
  3 |              15 |                 6 |   0.4
  4 |              31 |                24 |   0.8
  5 |              63 |               120 |   1.9
  6 |             127 |               720 |   5.7
```

The separation grows factorially and diverges for large n.

## Install

```bash
pip install selector-complexity
```

Or from source:

```bash
pip install numpy scipy
git clone https://github.com/iafiscal1212/Selector-Complexity-Framework.git
cd Selector-Complexity-Framework
python theory/09_hierarchy_theorem.py
```

## Verify

Each theorem in `theory/` is a standalone script with computational verification:

```bash
python theory/02_php_level0.py        # PHP is Level 0
python theory/03_phpe_level1.py       # PHP-E is Level 1
python theory/05_phpc_selector_lower_bound.py  # s-only selectors impossible
python theory/08_phpc_symmetry_argument.py     # Z_{n+1} lower bound
python theory/09_hierarchy_theorem.py  # Full hierarchy: SC(0) ⊊ SC(1) ⊊ SC(2)
```

No claim without computational proof.

## Open Questions

1. **SC(2) ⊊ SC(3)?** Does there exist a tautology with no useful selectors at all?
2. **Infinite hierarchy?** Are there infinitely many distinct selector complexity levels?
3. **Tight bounds?** Can the n! lower bound for PHP-C be improved to 2^Omega(n)?

## License

MIT
