#!/usr/bin/env python3
"""
Algorithmic pattern extractor from discovered selectors.

Reconstructs selector polynomials from result metadata, analyzes
which variables they use, and identifies algorithmic patterns
(DP rediscovery, structural decomposition, etc.)

Usage:
    python analysis/extract_patterns.py

Author: Carmen Esteban
"""

import os
import sys
import json
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

from discovery.run_discovery import build_job


def load_results():
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            results.append(json.load(f))
    return results


def parse_system(name):
    """'PHP-E(3)' -> ('phpe', 3), mapping to build_job types."""
    type_map = {
        "PHP": "php", "PHP-E": "phpe", "PHP-C": "phpc",
        "Tseitin": "tseitin", "3-XOR": "kxor", "SubsetSum": "subset_sum",
        "Factoring": "factoring", "Goldreich": "goldreich", "BinLWE": "lwe",
    }
    family = name[:name.index("(")]
    param = int(name[name.index("(") + 1:name.index(")")])
    return type_map.get(family, family), param, family


def reconstruct_selectors(source, var_maps, num_vars):
    """Reconstruct product selectors from the source field.

    E.g. source='template:product:vars6_7' -> product selectors over vars [6, 7].
    """
    if "template:product:vars" not in source:
        return None

    var_part = source.split("vars")[-1]
    var_indices = [int(v) for v in var_part.split("_")]

    # Build product selectors: for each bit pattern, product of v_i or (1-v_i)
    selectors = {}
    k = len(var_indices)
    for bits in range(2 ** k):
        terms = [(1.0, frozenset())]
        for i, v in enumerate(var_indices):
            if (bits >> i) & 1:
                factor = [(1.0, frozenset([v]))]
            else:
                factor = [(1.0, frozenset()), (-1.0, frozenset([v]))]
            # Multiply
            new_terms = {}
            for c1, m1 in terms:
                for c2, m2 in factor:
                    m = m1 | m2
                    new_terms[m] = new_terms.get(m, 0.0) + c1 * c2
            terms = [(c, m) for m, c in new_terms.items() if abs(c) > 1e-12]
        selectors[bits] = terms
    return selectors, var_indices


def classify_variable(var_idx, var_maps):
    """Determine if a variable index is primary (x) or auxiliary, and which type."""
    for key, mapping in var_maps.items():
        for label, idx in mapping.items():
            if idx == var_idx:
                return key, label
    return "unknown", var_idx


def analyze_system(result):
    """Full analysis of a single system's discovered selectors."""
    system = result["system"]
    best = result.get("best_selector")
    if not best or not best.get("source"):
        return None

    source = best["source"]
    stype, param, family = parse_system(system)

    # Rebuild the system to get var_maps
    try:
        axioms, num_vars, var_maps = build_job(stype, param)
    except Exception as e:
        return {"system": system, "error": str(e)}

    # Check if selectors were already serialized
    if "polynomials" in best and best["polynomials"]:
        selectors = best["polynomials"]
        var_indices = set()
        for label, terms in selectors.items():
            for t in terms:
                var_indices.update(t["m"])
        var_indices = sorted(var_indices)
    else:
        # Reconstruct from source
        rec = reconstruct_selectors(source, var_maps, num_vars)
        if rec is None:
            return {"system": system, "source": source, "note": "non-product selector"}
        selectors, var_indices = rec

    # Classify the selector variables
    var_roles = {}
    for v in var_indices:
        var_type, label = classify_variable(v, var_maps)
        var_roles[v] = (var_type, label)

    # Determine primary vs auxiliary usage
    primary_vars = [v for v in var_indices if var_roles[v][0] == "var_x"]
    aux_vars = [(v, var_roles[v]) for v in var_indices if var_roles[v][0] != "var_x"]

    # Pattern identification
    patterns = []

    if aux_vars:
        aux_types = set(role for _, (role, _) in aux_vars)
        for atype in aux_types:
            vars_of_type = [(v, lbl) for v, (r, lbl) in aux_vars if r == atype]
            labels = [lbl for _, lbl in vars_of_type]

            if atype == "var_d":
                # DP table variables — check if they form a contiguous DP row/column
                if all(isinstance(lbl, tuple) and len(lbl) == 2 for lbl in labels):
                    rows = set(lbl[0] for lbl in labels)
                    cols = set(lbl[1] for lbl in labels)
                    patterns.append({
                        "type": "dp_table_partition",
                        "description": f"Selectors partition by DP cell values "
                                       f"(rows {sorted(rows)}, cols {sorted(cols)})",
                        "algorithmic_meaning": "Rediscovers dynamic programming structure — "
                                               "the DP table cells serve as natural branch points"
                    })

            elif atype == "var_c":
                # Subset-Sum DP cells
                if all(isinstance(lbl, tuple) for lbl in labels):
                    patterns.append({
                        "type": "dp_table_partition",
                        "description": f"Selectors use DP cells from Subset-Sum table",
                        "algorithmic_meaning": "DP table provides efficient case decomposition"
                    })

            elif atype == "var_z":
                # Factoring partial products
                if all(isinstance(lbl, tuple) and len(lbl) == 2 for lbl in labels):
                    patterns.append({
                        "type": "partial_product_partition",
                        "description": f"Selectors partition by partial products z_{{i,j}}",
                        "algorithmic_meaning": "Grade-school multiplication structure — "
                                               "decomposition by digit products"
                    })

            elif atype == "var_a":
                # Goldreich AND gates
                patterns.append({
                    "type": "gate_partition",
                    "description": f"Selectors partition by AND gate outputs",
                    "algorithmic_meaning": "Circuit gate values as branch points — "
                                           "decomposes by local predicate evaluations"
                })

            elif atype == "var_y":
                patterns.append({
                    "type": "extension_partition",
                    "description": f"Selectors use PHP extension variables",
                    "algorithmic_meaning": "LPI (Last Pigeon Indicator) — "
                                           "decomposition by which pigeon is unmapped"
                })

            elif atype == "var_s":
                patterns.append({
                    "type": "counting_partition",
                    "description": f"Selectors use PHP counting variables",
                    "algorithmic_meaning": "Counting argument structure — "
                                           "branch by counting variable values"
                })

    if not aux_vars and primary_vars:
        patterns.append({
            "type": "primary_partition",
            "description": f"Selectors use only primary variables {primary_vars}",
            "algorithmic_meaning": "Direct case split on input bits — no auxiliary structure exploited"
        })

    # Check if selector was discovered via axiom graph decomposition
    if "axiom_graph" in source:
        substrategy = source.split(":")[1] if ":" in source else "unknown"
        patterns.append({
            "type": f"graph_{substrategy}",
            "description": f"Discovered via axiom graph {substrategy} analysis",
            "algorithmic_meaning": {
                "hinge": "Variable connects different axiom communities — "
                         "structural fault line in the proof topology",
                "bridge": "Variable has high betweenness in co-occurrence graph — "
                          "bridges otherwise-separated variable clusters",
                "bottleneck": "Variable removal maximally fragments axiom graph — "
                              "carries the proof's structural weight",
            }.get(substrategy,
                  "Axiom graph topology reveals natural selector variable")
        })

    return {
        "system": system,
        "family": family,
        "param": param,
        "sc_level": result.get("level_estimate"),
        "source": source,
        "selector_vars": var_indices,
        "var_roles": {str(v): {"type": r, "label": str(l)} for v, (r, l) in var_roles.items()},
        "num_primary": len(primary_vars),
        "num_auxiliary": len(aux_vars),
        "num_selectors": best.get("num_selectors", "?"),
        "patterns": patterns,
    }


def main():
    results = load_results()
    print(f"Loaded {len(results)} results\n")

    analyses = []
    for r in results:
        a = analyze_system(r)
        if a:
            analyses.append(a)

    # Print per-system analysis
    print("=" * 70)
    print("SELECTOR VARIABLE ANALYSIS")
    print("=" * 70)

    for a in analyses:
        if "error" in a:
            print(f"\n{a['system']}: ERROR - {a['error']}")
            continue
        if "note" in a:
            print(f"\n{a['system']}: {a['note']} ({a['source']})")
            continue

        print(f"\n--- {a['system']} (SC({a['sc_level']})) ---")
        print(f"  Source: {a['source']}")
        print(f"  Selector vars: {a['selector_vars']}")
        print(f"  Roles: {a['num_primary']} primary, {a['num_auxiliary']} auxiliary")

        for v, info in a["var_roles"].items():
            print(f"    var {v}: {info['type']} [{info['label']}]")

        for p in a.get("patterns", []):
            print(f"  PATTERN: {p['type']}")
            print(f"    {p['description']}")
            print(f"    -> {p['algorithmic_meaning']}")

    # Cross-family pattern summary
    print("\n" + "=" * 70)
    print("CROSS-FAMILY PATTERN SUMMARY")
    print("=" * 70)

    pattern_families = defaultdict(list)
    for a in analyses:
        for p in a.get("patterns", []):
            pattern_families[p["type"]].append(a["family"])

    for ptype, families in sorted(pattern_families.items()):
        unique = sorted(set(families))
        print(f"\n{ptype}: found in {', '.join(unique)}")

    # Key insight: DP table families
    dp_families = [a for a in analyses
                   if any(p["type"] == "dp_table_partition" for p in a.get("patterns", []))]
    if dp_families:
        print("\n" + "-" * 40)
        print("KEY INSIGHT: DP-based decomposition")
        print("-" * 40)
        for a in dp_families:
            print(f"  {a['system']}: SC({a['sc_level']}) — "
                  f"{a['num_auxiliary']} aux vars from DP table")
        scs = [a["sc_level"] for a in dp_families]
        print(f"  SC range: {min(scs)}-{max(scs)}")
        print(f"  -> DP table structure provides natural selectors that")
        print(f"     reduce certificate complexity across different problems")

    # Save full analysis
    out_path = os.path.join(PROJECT_ROOT, "analysis", "pattern_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
