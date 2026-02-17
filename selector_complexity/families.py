"""
Tautology Family Builders
==========================

Builders for cryptographic/combinatorial tautology families used in the
SC landscape: Factoring, Goldreich PRG, and Binary LWE.

Each builder returns (axioms, num_vars, ...) where axioms is a list of
polynomial constraints in multilinear form.

Functions:
    factoring_axioms       -- integer factoring (prime > 2^n)
    goldreich_prg_axioms   -- Goldreich PRG inversion (P5 predicate)
    binary_lwe_axioms      -- binary LWE with Hamming weight bound

Author: Carmen Esteban
License: MIT
"""

import random
from itertools import combinations


# =====================================================================
# FACTORING
# =====================================================================

def factoring_axioms(n, seed=42):
    """Build unsatisfiable factoring instance: find P,Q of n bits with P*Q = N.

    N is chosen as a prime > 2^n, so P*Q = N requires {P,Q} = {1,N},
    but N doesn't fit in n bits -> unsatisfiable.

    Variables:
      - var_p[i]: bit i of P  (n primary vars)
      - var_q[j]: bit j of Q  (n primary vars)
      - var_z[(i,j)]: partial product p_i * q_j  (n^2 auxiliary vars)
      Total: 2n + n^2

    Axioms:
      a) z_{i,j} - p_i * q_j = 0          (n^2 axioms, degree 2)
      b) sum_{i,j} 2^{i+j} * z_{i,j} - N = 0  (1 axiom, linear in z)

    Returns (axioms, num_vars, var_p, var_q, var_z).
    """
    def _next_prime_above(x):
        c = x + 1 if x % 2 == 0 else x + 2
        while True:
            if all(c % d != 0 for d in range(2, int(c**0.5) + 1)):
                return c
            c += 2
    N = _next_prime_above(2 ** n)

    idx = 0
    var_p = {}
    for i in range(n):
        var_p[i] = idx
        idx += 1

    var_q = {}
    for j in range(n):
        var_q[j] = idx
        idx += 1

    var_z = {}
    for i in range(n):
        for j in range(n):
            var_z[(i, j)] = idx
            idx += 1
    num_vars = idx

    axioms = []

    # (a) z_{i,j} - p_i * q_j = 0
    for i in range(n):
        for j in range(n):
            axioms.append([
                (1.0, frozenset([var_z[(i, j)]])),
                (-1.0, frozenset([var_p[i], var_q[j]])),
            ])

    # (b) sum_{i,j} 2^{i+j} * z_{i,j} - N = 0
    target_terms = [(-float(N), frozenset())]
    for i in range(n):
        for j in range(n):
            coeff = float(2 ** (i + j))
            target_terms.append((coeff, frozenset([var_z[(i, j)]])))
    axioms.append(target_terms)

    return axioms, num_vars, var_p, var_q, var_z


# =====================================================================
# GOLDREICH PRG
# =====================================================================

def goldreich_prg_axioms(n, stretch=2, k=5, seed=42):
    """Build unsatisfiable Goldreich PRG inversion instance.

    Uses predicate P5: y_j = x_{s1} XOR (x_{s2}*x_{s3}) XOR (x_{s4}*x_{s5}).
    Plants a seed, computes output, flips last bit -> contradiction.

    Variables:
      - var_x[i]: seed bits  (n primary vars)
      - var_a[(j,0)], var_a[(j,1)]: AND gate outputs  (2m auxiliary vars)
      Total: n + 2m where m = stretch * n

    Axioms per output bit j:
      a) a_{j,0} - x_{s2}*x_{s3} = 0
      b) a_{j,1} - x_{s4}*x_{s5} = 0
      c) XOR polynomial: x_{s1} + a_{j,0} + a_{j,1}
         - 2*x_{s1}*a_{j,0} - 2*x_{s1}*a_{j,1} - 2*a_{j,0}*a_{j,1}
         + 4*x_{s1}*a_{j,0}*a_{j,1} - y_j = 0

    Returns (axioms, num_vars, var_x, var_a).
    """
    rng = random.Random(seed)
    m = stretch * n

    idx = 0
    var_x = {}
    for i in range(n):
        var_x[i] = idx
        idx += 1

    var_a = {}
    for j in range(m):
        var_a[(j, 0)] = idx
        idx += 1
        var_a[(j, 1)] = idx
        idx += 1
    num_vars = idx

    neighborhoods = []
    for j in range(m):
        nbrs = sorted(rng.sample(range(n), min(k, n)))
        neighborhoods.append(nbrs)

    planted = [rng.randint(0, 1) for _ in range(n)]
    output = []
    for j in range(m):
        s = neighborhoods[j]
        s1 = planted[s[0]]
        and1 = planted[s[1]] * planted[s[2]] if len(s) > 2 else 0
        and2 = planted[s[3]] * planted[s[4]] if len(s) > 4 else 0
        y_j = s1 ^ and1 ^ and2
        output.append(y_j)

    output[-1] ^= 1

    axioms = []

    for j in range(m):
        s = neighborhoods[j]
        xs1 = var_x[s[0]]
        xs2 = var_x[s[1]] if len(s) > 1 else var_x[s[0]]
        xs3 = var_x[s[2]] if len(s) > 2 else var_x[s[0]]
        xs4 = var_x[s[3]] if len(s) > 3 else var_x[s[0]]
        xs5 = var_x[s[4]] if len(s) > 4 else var_x[s[0]]

        aj0 = var_a[(j, 0)]
        aj1 = var_a[(j, 1)]
        y_j = output[j]

        axioms.append([
            (1.0, frozenset([aj0])),
            (-1.0, frozenset([xs2, xs3])),
        ])

        axioms.append([
            (1.0, frozenset([aj1])),
            (-1.0, frozenset([xs4, xs5])),
        ])

        axioms.append([
            (-float(y_j), frozenset()),
            (1.0, frozenset([xs1])),
            (1.0, frozenset([aj0])),
            (1.0, frozenset([aj1])),
            (-2.0, frozenset([xs1, aj0])),
            (-2.0, frozenset([xs1, aj1])),
            (-2.0, frozenset([aj0, aj1])),
            (4.0, frozenset([xs1, aj0, aj1])),
        ])

    return axioms, num_vars, var_x, var_a


# =====================================================================
# BINARY LWE HELPERS
# =====================================================================

def _find_lwe_instance(n, m, t, start_seed, max_seeds=500):
    """Search for a binary matrix A and vector b giving an UNSAT LWE instance.

    Tries successive seeds for A until finding one where the covering
    radius of the code {A*s mod 2} exceeds t.  Returns (A, b).
    """
    for seed in range(start_seed, start_seed + max_seeds):
        rng = random.Random(seed)
        A = [[rng.randint(0, 1) for _ in range(n)] for _ in range(m)]

        all_syndromes = set()
        for s_bits in range(2 ** n):
            s_vec = [(s_bits >> j) & 1 for j in range(n)]
            syndrome = tuple(
                sum(A[i][j] * s_vec[j] for j in range(n)) % 2
                for i in range(m))
            all_syndromes.add(syndrome)

        rng_b = random.Random(seed + 10000)
        best_b = None
        best_min_weight = -1
        n_candidates = min(5000, 2 ** m)

        for _ in range(n_candidates):
            b_cand = tuple(rng_b.randint(0, 1) for _ in range(m))
            min_w = m + 1
            for syn in all_syndromes:
                w = sum(b_cand[i] ^ syn[i] for i in range(m))
                min_w = min(min_w, w)
                if min_w <= t:
                    break
            if min_w > best_min_weight:
                best_min_weight = min_w
                best_b = b_cand

        if best_min_weight > t:
            return A, list(best_b)

    return None, None


# =====================================================================
# BINARY LWE
# =====================================================================

def binary_lwe_axioms(n, m_factor=2, error_bound=None, seed=42):
    """Build unsatisfiable binary LWE instance: A*s + e = b (mod 2).

    Generates random binary matrix A, then finds b such that for all
    secrets s, the error e = b - A*s (mod 2) has Hamming weight > t.

    Variables:
      - var_s[j]: secret bits  (n primary vars)
      - var_e[i]: error bits   (m primary vars)
      - var_d[(i,k)]: DP table for Hamming weight bound  ((m+1)*(t+1) auxiliary)
      Total: n + m + (m+1)*(t+1)

    Axioms:
      a) Sample XOR: for each row i of A, XOR encoding of A[i]*s + e_i = b_i
      b) DP Hamming weight recurrence
      c) Feasibility: d_{m,0} + d_{m,1} + ... + d_{m,t} = 1

    Returns (axioms, num_vars, var_s, var_e, var_d).
    """
    m = m_factor * n
    t = error_bound if error_bound is not None else max(1, m // 4)

    A, b = _find_lwe_instance(n, m, t, seed)

    assert b is not None, (
        "binary_lwe_axioms: could not find UNSAT instance for n={}, t={}".format(n, t)
    )

    idx = 0
    var_s = {}
    for j in range(n):
        var_s[j] = idx
        idx += 1

    var_e = {}
    for i in range(m):
        var_e[i] = idx
        idx += 1

    var_d = {}
    for i in range(m + 1):
        for k in range(t + 1):
            var_d[(i, k)] = idx
            idx += 1
    num_vars = idx

    axioms = []

    # (a) Sample XOR constraints
    for i in range(m):
        xor_vars = [var_s[j] for j in range(n) if A[i][j] == 1]
        xor_vars.append(var_e[i])
        b_i = b[i]

        kk = len(xor_vars)
        terms = [(-float(b_i), frozenset())]
        for j_idx in range(1, kk + 1):
            coeff = (-2.0) ** (j_idx - 1)
            for subset in combinations(xor_vars, j_idx):
                terms.append((coeff, frozenset(subset)))
        axioms.append(terms)

    # (b) DP Hamming weight recurrence
    axioms.append([
        (-1.0, frozenset()),
        (1.0, frozenset([var_d[(0, 0)]])),
    ])
    for k in range(1, t + 1):
        axioms.append([
            (1.0, frozenset([var_d[(0, k)]])),
        ])

    for i in range(1, m + 1):
        ei = var_e[i - 1]
        for k in range(t + 1):
            terms = [
                (1.0, frozenset([var_d[(i, k)]])),
                (-1.0, frozenset([var_d[(i - 1, k)]])),
                (1.0, frozenset([ei, var_d[(i - 1, k)]])),
            ]
            if k - 1 >= 0:
                terms.append((-1.0, frozenset([ei, var_d[(i - 1, k - 1)]])))
            axioms.append(terms)

    # (c) Feasibility
    feas_terms = [(-1.0, frozenset())]
    for k in range(t + 1):
        feas_terms.append((1.0, frozenset([var_d[(m, k)]])))
    axioms.append(feas_terms)

    return axioms, num_vars, var_s, var_e, var_d
