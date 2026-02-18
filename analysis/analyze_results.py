#!/usr/bin/env python3
"""
Analyze discovery results: SC scaling, cross-family comparison,
and selector structure patterns.

Usage:
    python analysis/analyze_results.py           # table + scaling
    python analysis/analyze_results.py --full    # + selector structure analysis

Author: Carmen Esteban
"""

import os
import sys
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_results():
    """Load all result JSONs into a list of dicts."""
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        # Infer family and parameter from system name
        system = data.get("system", fname.replace(".json", ""))
        family, param = _parse_system_name(system)
        data["_family"] = family
        data["_param"] = param
        data["_file"] = fname
        results.append(data)
    return results


def _parse_system_name(name):
    """Parse 'PHP-E(3)' into ('PHP-E', 3)."""
    if "(" in name and ")" in name:
        family = name[:name.index("(")]
        param = name[name.index("(") + 1:name.index(")")]
        try:
            param = int(param)
        except ValueError:
            pass
        return family, param
    return name, None


# ---------------------------------------------------------------
# 1. SC Scaling Table
# ---------------------------------------------------------------

def print_scaling_table(results):
    """Print SC level per family and parameter."""
    families = defaultdict(list)
    for r in results:
        families[r["_family"]].append(r)

    # Sort families in a meaningful order
    family_order = [
        "PHP", "PHP-E", "PHP-C", "Tseitin", "3-XOR",
        "SubsetSum", "Factoring", "Goldreich", "BinLWE",
    ]
    all_families = sorted(families.keys(),
                          key=lambda f: family_order.index(f) if f in family_order else 99)

    print("=" * 70)
    print("SC SCALING TABLE")
    print("=" * 70)
    print(f"{'Family':<14} {'n':>4}  {'SC':>4}  {'Vars':>6}  {'Axioms':>7}  "
          f"{'Cert':>8}  {'w/Sel':>8}  {'Source'}")
    print("-" * 70)

    for family in all_families:
        entries = sorted(families[family], key=lambda r: r["_param"])
        for r in entries:
            sc = r.get("level_estimate", "?")
            n_vars = r.get("n", "?")
            best = r.get("best_selector", {})
            cert_base = r.get("cert_without_selectors", {})
            cert_sel = r.get("cert_with_selectors", {})
            source = best.get("source", "-")
            # Count axioms from candidates_tested
            axioms = r.get("candidates_tested", "?")

            cert_size = cert_base.get("size", "-") if cert_base else "-"
            cert_sel_size = cert_sel.get("size", "-") if cert_sel else "-"

            print(f"{r['system']:<14} {r['_param']:>4}  "
                  f"SC({sc})  {n_vars:>6}  {axioms:>7}  "
                  f"{cert_size:>8}  {cert_sel_size:>8}  {source}")
        print()


# ---------------------------------------------------------------
# 2. Cross-family comparison
# ---------------------------------------------------------------

def print_cross_family(results):
    """Group families by SC level and compare."""
    by_sc = defaultdict(list)
    for r in results:
        sc = r.get("level_estimate", -1)
        by_sc[sc].append(r)

    print("=" * 70)
    print("CROSS-FAMILY COMPARISON (grouped by SC level)")
    print("=" * 70)

    for sc in sorted(by_sc.keys()):
        entries = by_sc[sc]
        families = set(r["_family"] for r in entries)
        print(f"\nSC({sc}): {len(entries)} instances from {len(families)} families")
        print(f"  Families: {', '.join(sorted(families))}")

        # Check for selector structure patterns
        sources = defaultdict(int)
        for r in entries:
            best = r.get("best_selector", {})
            src = best.get("source", "none")
            # Simplify source to strategy name
            strategy = src.split(":")[0] if ":" in src else src
            sources[strategy] += 1

        print(f"  Selector sources: ", end="")
        print(", ".join(f"{s}({c})" for s, c in sorted(sources.items(), key=lambda x: -x[1])))

        # Certificate compression ratio
        ratios = []
        for r in entries:
            c_base = r.get("cert_without_selectors", {})
            c_sel = r.get("cert_with_selectors", {})
            if c_base and c_sel and c_base.get("size") and c_sel.get("size"):
                ratio = c_sel["size"] / c_base["size"]
                ratios.append(ratio)
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"  Avg cert compression: {avg:.2%} of baseline")


# ---------------------------------------------------------------
# 3. Selector structure analysis
# ---------------------------------------------------------------

def print_selector_analysis(results):
    """Analyze what makes selectors work for each family."""
    print("\n" + "=" * 70)
    print("SELECTOR STRUCTURE ANALYSIS")
    print("=" * 70)

    families = defaultdict(list)
    for r in results:
        families[r["_family"]].append(r)

    for family, entries in sorted(families.items()):
        entries.sort(key=lambda r: r["_param"])
        print(f"\n--- {family} ---")

        scs = [r.get("level_estimate", "?") for r in entries]
        params = [r["_param"] for r in entries]

        # SC trend
        if len(set(scs)) == 1:
            print(f"  SC constant at SC({scs[0]}) across n={params}")
        else:
            transitions = []
            for i in range(1, len(scs)):
                if scs[i] != scs[i - 1]:
                    transitions.append(f"SC({scs[i-1]})→SC({scs[i]}) at n={params[i]}")
            if transitions:
                print(f"  SC transitions: {'; '.join(transitions)}")
            print(f"  SC sequence: {' → '.join(f'SC({s})' for s in scs)} for n={params}")

        # Best selector characterization
        for r in entries:
            best = r.get("best_selector", {})
            if best:
                src = best.get("source", "none")
                k = best.get("num_selectors", "?")
                deg = best.get("degree", "?")
                print(f"  n={r['_param']}: k={k} selectors, deg={deg}, source={src}")


# ---------------------------------------------------------------
# 4. Algorithmic pattern detection
# ---------------------------------------------------------------

def print_algorithmic_patterns(results):
    """Look for algorithmic patterns in selector families."""
    print("\n" + "=" * 70)
    print("ALGORITHMIC PATTERN DETECTION")
    print("=" * 70)

    # Group DP-based families (Subset-Sum, Binary LWE)
    dp_families = [r for r in results if r["_family"] in ("SubsetSum", "BinLWE")]
    if dp_families:
        print("\n[DP-based families: SubsetSum, BinLWE]")
        for r in sorted(dp_families, key=lambda x: (x["_family"], x["_param"])):
            sc = r.get("level_estimate", "?")
            best = r.get("best_selector", {})
            print(f"  {r['system']}: SC({sc}), best={best.get('source', '-')}")
        scs_ss = [r.get("level_estimate") for r in dp_families if r["_family"] == "SubsetSum"]
        scs_lwe = [r.get("level_estimate") for r in dp_families if r["_family"] == "BinLWE"]
        if scs_ss and scs_lwe:
            print(f"  SubsetSum SC range: {min(scs_ss)}-{max(scs_ss)}")
            print(f"  BinLWE SC range: {min(scs_lwe)}-{max(scs_lwe)}")
            if set(scs_ss) & set(scs_lwe):
                print(f"  → Shared SC levels suggest shared algorithmic structure (DP tables)")

    # Crypto hardness families (Factoring, Goldreich)
    crypto_families = [r for r in results if r["_family"] in ("Factoring", "Goldreich")]
    if crypto_families:
        print("\n[Crypto hardness families: Factoring, Goldreich]")
        for r in sorted(crypto_families, key=lambda x: (x["_family"], x["_param"])):
            sc = r.get("level_estimate", "?")
            best = r.get("best_selector", {})
            print(f"  {r['system']}: SC({sc}), best={best.get('source', '-')}")

    # Auxiliary variable impact
    print("\n[Auxiliary variable impact on SC]")
    aux_families = {
        "PHP-E": "var_y (extension)",
        "PHP-C": "var_s (counting)",
        "SubsetSum": "var_c (DP table)",
        "Factoring": "var_z (partial products)",
        "Goldreich": "var_a (AND gates)",
        "BinLWE": "var_d (Hamming DP)",
    }
    for family, aux_desc in aux_families.items():
        entries = [r for r in results if r["_family"] == family]
        if entries:
            scs = [r.get("level_estimate", "?") for r in entries]
            print(f"  {family} ({aux_desc}): SC = {', '.join(str(s) for s in scs)}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze discovery results")
    parser.add_argument("--full", action="store_true",
                        help="Include selector structure and pattern analysis")
    args = parser.parse_args()

    results = load_results()
    if not results:
        print(f"No results found in {RESULTS_DIR}")
        return

    print(f"Loaded {len(results)} results from {RESULTS_DIR}\n")

    print_scaling_table(results)
    print_cross_family(results)

    if args.full:
        print_selector_analysis(results)
        print_algorithmic_patterns(results)


if __name__ == "__main__":
    main()
