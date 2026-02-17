# Selector Complexity: A Framework for IPS Certificate Hardness

**Author:** Carmen Esteban
**Date:** February 2026
**Status:** In progress

## What is this?

A formal framework that classifies tautologies by the efficiency of their
"selector" polynomials in Ideal Proof Systems (IPS).

## Structure

```
Selector-Complexity-Framework/
├── theory/
│   ├── 01_definitions.py          # Formal definitions + computational verification
│   ├── 02_php_level0.py           # PROOF: PHP is Level 0 (no selectors needed)
│   ├── 03_phpe_level1.py          # PROOF: PHP-E is Level 1 (efficient selectors)
│   ├── 04_phpc_level2_conjecture.py  # CONJECTURE: PHP-C is Level 2+
│   └── 05_hierarchy_theorem.py    # PROOF: Level k < Level k+1 (strict hierarchy)
├── paper/
│   ├── selector_complexity.tex    # Paper draft
│   └── figures/                   # Generated figures
├── tests/
│   ├── test_definitions.py        # Verify all definitions
│   ├── test_php_level0.py         # Verify PHP classification
│   ├── test_phpe_level1.py        # Verify PHP-E classification
│   └── run_all_tests.py           # Run everything
└── results/
    └── *.json                     # Computational results
```

## Selector Complexity Levels

| Level | Description | Example | IPS Size | Status |
|-------|-------------|---------|----------|--------|
| 0 | No selectors needed | PHP | O(n^2) | PROVEN |
| 1 | Efficient selectors exist | PHP-E | O(n^4) | PROVEN |
| 2 | Only exponential selectors | PHP-C? | 2^poly(n)? | CONJECTURE |
| 3 | No useful selectors | ??? | 2^Omega(n) | OPEN |

## How to verify

```bash
pip install numpy scipy
cd tests
python run_all_tests.py
```

Every theorem has a computational verification. No claim without proof.
