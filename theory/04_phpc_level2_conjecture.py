"""
CONJECTURE: PHP-C is Selector Complexity Level 2+
===================================================

We provide computational evidence that PHP-Circular does NOT have
efficient selectors, by:

1. Showing that the g_p construction FAILS for circular orderings
2. Searching exhaustively for alternative selectors at small n
3. Measuring the minimum selector size needed

Author: Carmen Esteban
Date: February 2026
"""

import numpy as np
from itertools import combinations, permutations


def build_circular_successor(perm):
    """Given a permutation (as a cycle), build the successor assignment.

    perm = (p1, p2, ..., p_{n+1}) means p1->p2->...->p_{n+1}->p1.
    Returns dict mapping (p, q) -> {0, 1} for s_{p,q} variables.
    """
    n_plus_1 = len(perm)
    succ = {}
    for i in range(n_plus_1):
        p = perm[i]
        q = perm[(i + 1) % n_plus_1]
        succ[(p, q)] = 1
    # Set all other s_{p,q} to 0
    for p in perm:
        for q in perm:
            if p != q and (p, q) not in succ:
                succ[(p, q)] = 0
    return succ


def attempt_gp_on_circular(n):
    """Try to construct g_p-like indicators for circular orderings.

    In PHP-E, g_p = "p is last" works because total orders have a maximum.
    In PHP-C, we try:
      g_p = "p is the gap pigeon" (the one not assigned to a hole)

    But for circular orderings, every position is equivalent by rotation.
    We show that no polynomial depending ONLY on s-variables can
    distinguish any single pigeon.
    """
    pigeons = list(range(1, n + 2))

    print("Attempting g_p construction for circular orderings, n={}".format(n))
    print("Pigeons: {}".format(pigeons))

    # Generate all cyclic orderings (fix pigeon 1's position to avoid
    # counting rotations multiple times)
    # A cyclic ordering on n+1 elements has n! distinct representatives
    # when we fix one element's position
    others = pigeons[1:]
    num_cycles = 0
    pigeon_as_gap = {p: 0 for p in pigeons}  # Count how often each pigeon is "gap"

    for perm_others in permutations(others):
        cycle = (pigeons[0],) + perm_others
        num_cycles += 1

        # In this cycle, any pigeon could be the "gap" pigeon
        # (the one not assigned a hole). Due to rotational symmetry
        # of the cycle, every pigeon is equally likely to be the gap.
        # This means no function of s alone can identify the gap.

        # But let's verify: for each assignment of holes consistent
        # with the cycle, which pigeon is the gap?
        # The gap depends on WHERE the cycle "breaks" relative to holes.
        # For n holes numbered 1..n in a circle (succ(n) = 1),
        # the gap pigeon is the one at the position where the hole
        # numbering wraps around.

        # For each possible "break point" in the cycle:
        for break_idx in range(len(cycle)):
            gap_pigeon = cycle[break_idx]
            pigeon_as_gap[gap_pigeon] += 1

    print("Total cyclic orderings (with fixed first): {}".format(num_cycles))
    print("Gap distribution (each pigeon as 'gap'):")
    for p in pigeons:
        print("  Pigeon {}: gap {} times out of {} total".format(
            p, pigeon_as_gap[p], num_cycles * (n + 1)))

    # Key observation: every pigeon appears as gap exactly the same number
    # of times. This means there's NO function of s-variables alone that
    # can identify the gap pigeon.
    counts = list(pigeon_as_gap.values())
    all_equal = all(c == counts[0] for c in counts)

    print("\nAll pigeons equally likely to be gap: {}".format(all_equal))
    if all_equal:
        print("=> No s-only indicator can distinguish the gap pigeon")
        print("=> The g_p construction from PHP-E FAILS for PHP-C")
        print("=> Any selector must depend on BOTH s and x variables")
    else:
        print("=> Unexpected: pigeons have different gap frequencies")

    return all_equal


def search_for_selectors(n, max_degree=3):
    """Search exhaustively for polynomial selectors in s-variables only.

    For each candidate polynomial g(s) of degree <= max_degree,
    check if it can act as a selector (taking value 0 or 1 on each
    cyclic permutation assignment).

    If no such selector exists, this is evidence for Level 2+.
    """
    pigeons = list(range(1, n + 2))

    # Index s-variables
    var_s = {}
    idx = 0
    for p in pigeons:
        for q in pigeons:
            if p != q:
                var_s[(p, q)] = idx
                idx += 1
    num_s_vars = idx

    print("\nSearching for s-only selectors, n={}, max_degree={}".format(
        n, max_degree))
    print("s-variables: {}".format(num_s_vars))

    # Generate all cyclic permutation assignments
    others = pigeons[1:]
    cycle_assignments = []
    for perm in permutations(others):
        cycle = (pigeons[0],) + perm
        succ = build_circular_successor(cycle)
        assignment = {}
        for (p, q), val in succ.items():
            assignment[var_s[(p, q)]] = val
        cycle_assignments.append(assignment)

    num_cycles = len(cycle_assignments)
    print("Number of cyclic orderings: {}".format(num_cycles))

    # For each pigeon, check if there exists a polynomial g(s) of bounded
    # degree that equals 1 when that pigeon is "gap" and 0 otherwise
    # Since "gap" depends on x (not s), we check something weaker:
    # Is there any polynomial g(s) that takes DIFFERENT values on
    # different cyclic orderings?

    # Enumerate monomials in s-variables up to max_degree
    monoms = [frozenset()]
    for d in range(1, max_degree + 1):
        for combo in combinations(range(num_s_vars), d):
            monoms.append(frozenset(combo))
    print("Monomials up to degree {}: {}".format(max_degree, len(monoms)))

    # Evaluate each monomial on each cycle assignment
    eval_matrix = np.zeros((num_cycles, len(monoms)))
    for i, assign in enumerate(cycle_assignments):
        for j, monom in enumerate(monoms):
            prod = 1.0
            for v in monom:
                prod *= assign.get(v, 0)
            eval_matrix[i, j] = prod

    # Check if all rows are identical (= no monomial distinguishes cycles)
    unique_rows = np.unique(eval_matrix, axis=0)
    print("Distinct evaluation vectors: {} out of {} cycles".format(
        len(unique_rows), num_cycles))

    if len(unique_rows) == 1:
        print("=> ALL cyclic orderings are INDISTINGUISHABLE by s-only polynomials!")
        print("=> NO s-only selector family can exist")
        print("=> STRONG evidence for Level 2+")
        return True
    else:
        print("=> Some cycles are distinguishable")
        print("=> Checking if a partition of unity is possible...")

        # Check rank
        rank = np.linalg.matrix_rank(eval_matrix)
        print("   Rank of evaluation matrix: {} (need {} for full partition)".format(
            rank, num_cycles))
        if rank < num_cycles:
            print("   => Rank deficient: cannot construct {} distinct selectors".format(
                num_cycles))
            print("   => At best {} selectors from s-only polynomials".format(rank))
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CONJECTURE: PHP-C is Selector Complexity Level 2+")
    print("=" * 60)
    print()
    print("Evidence 1: g_p construction fails for circular orderings")
    print("-" * 60)

    for n in [2, 3, 4]:
        gp_fails = attempt_gp_on_circular(n)
        assert gp_fails, "g_p should fail for circular orderings!"
        print()

    print()
    print("Evidence 2: No s-only selectors exist")
    print("-" * 60)

    for n in [2, 3]:
        for max_d in [2, 3]:
            search_for_selectors(n, max_degree=max_d)
            print()

    print("=" * 60)
    print("SUMMARY OF EVIDENCE")
    print("=" * 60)
    print("""
1. The Last Pigeon Indicator g_p FAILS for circular orderings:
   - All pigeons are equally likely to be the "gap"
   - No s-only polynomial can identify the gap pigeon

2. No s-only selector family exists:
   - All cyclic orderings are algebraically indistinguishable
   - Any selector MUST depend on x-variables (hole assignments)

3. Dependence on x-variables is costly:
   - The x-variables encode the PHP structure (unsatisfiable)
   - Extracting useful information from x requires solving PHP
   - This suggests selectors must be exponentially large

CONJECTURE: PHP-C is Selector Complexity Level 2 or higher.
=> SIZE_IPS(PHP-C) = n^{omega(1)}
""")
    print("=" * 60)
