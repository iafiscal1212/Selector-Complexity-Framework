"""
Discovery configuration and job definitions.
=============================================

Default search parameters and the queue of tautology instances to process.

Scaled config: PHP-E up to n=8, PHP-C up to n=5, PHP up to n=6.
Uses aip-engine AccordionBuilder for systems with 40+ variables.

Author: Carmen Esteban
"""

from selector_complexity.discovery import DiscoveryConfig


# --- Default search parameters ---

DEFAULT_CONFIG = DiscoveryConfig(
    max_degree=6,
    max_selector_count=20,
    max_selector_degree=4,
    strategies=[
        "exhaustive", "template", "ips_guided",
        "subspace_projection", "variable_group",
    ],
    max_vars_exhaustive=12,
    top_k_variables=10,
    monomial_cap=2_000_000,
    use_aip=True,
    verbose=True,
)


# --- Job queue ---
# Each entry: (system_name, builder_function_name, n)
# Ordered by increasing difficulty within each family.

JOB_QUEUE = [
    # --- PHP family: SC(0), cert directo ---
    ("PHP(2)",     "php",     2),
    ("PHP(3)",     "php",     3),
    ("PHP(4)",     "php",     4),
    ("PHP(5)",     "php",     5),
    ("PHP(6)",     "php",     6),

    # --- PHP-E family: SC(1), selectores LPI en y-variables ---
    ("PHP-E(2)",   "phpe",    2),
    ("PHP-E(3)",   "phpe",    3),
    ("PHP-E(4)",   "phpe",    4),
    ("PHP-E(5)",   "phpe",    5),
    ("PHP-E(6)",   "phpe",    6),
    ("PHP-E(7)",   "phpe",    7),
    ("PHP-E(8)",   "phpe",    8),

    # --- PHP-C family: SC(2), selectores s-variable no eficientes ---
    ("PHP-C(2)",   "phpc",    2),
    ("PHP-C(3)",   "phpc",    3),
    ("PHP-C(4)",   "phpc",    4),
    ("PHP-C(5)",   "phpc",    5),

    # --- Tseitin (ciclos, facil para IPS) ---
    ("Tseitin(6)", "tseitin", 6),
    ("Tseitin(8)", "tseitin", 8),

    # --- k-XOR: sin auxiliares, SC(0) vs SC(3) ---
    ("3-XOR(10)",    "kxor",       10),
    ("3-XOR(15)",    "kxor",       15),
    ("3-XOR(20)",    "kxor",       20),
    ("3-XOR(30)",    "kxor",       30),

    # --- Subset-Sum DP: auxiliares c_{i,s}, SC(1)/SC(2) ---
    ("SubsetSum(3)", "subset_sum",  3),
    ("SubsetSum(4)", "subset_sum",  4),
    ("SubsetSum(5)", "subset_sum",  5),

    # --- Factoring: auxiliar z_{i,j}, SC(2)/SC(3) ---
    ("Factoring(4)",   "factoring",  4),
    ("Factoring(6)",   "factoring",  6),
    ("Factoring(8)",   "factoring",  8),

    # --- Goldreich PRG: auxiliar a_{j}, SC(2)/SC(3) ---
    ("Goldreich(8)",   "goldreich",  8),
    ("Goldreich(12)",  "goldreich",  12),
    ("Goldreich(16)",  "goldreich",  16),

    # --- Binary LWE: auxiliar d_{i,k} (DP), SC(1)/SC(2) ---
    ("BinLWE(6)",      "lwe",        6),
    ("BinLWE(8)",      "lwe",        8),
    ("BinLWE(10)",     "lwe",        10),
]
