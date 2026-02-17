#!/usr/bin/env python3
"""
Selector Discovery Runner
==========================

Processes a queue of tautology instances, running the DiscoveryEngine
on each and saving results as JSON.

Usage:
    python discovery/run_discovery.py              # foreground
    python discovery/run_discovery.py --daemon      # background daemon
    python discovery/run_discovery.py --reset       # clear state and restart

Ctrl+C to stop cleanly between jobs.

Author: Carmen Esteban
"""

import os
import sys
import json
import time
import signal
import random
import argparse
from itertools import combinations, permutations

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from discovery.config import DEFAULT_CONFIG, JOB_QUEUE
from selector_complexity.discovery import DiscoveryEngine
from selector_complexity.php import php_axioms, phpe_axioms, phpc_axioms


# --- Paths ---

STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "state.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --- State management ---

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "current": None, "started_at": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# --- Tseitin formula builder ---

def tseitin_axioms(n):
    """Build unsatisfiable Tseitin tautology on a cycle graph with n vertices.

    Each vertex gets charge 1 except possibly the last one, adjusted so
    that the total charge is odd (guaranteeing unsatisfiability on any
    even or odd cycle).

    XOR constraint at vertex v: x_a + x_b - 2*x_a*x_b = charge(v)
    """
    edges = [(i, (i + 1) % n) for i in range(n)]

    # Variable x_e for each edge
    var_x = {}
    idx = 0
    for e in edges:
        var_x[e] = idx
        idx += 1
    num_vars = idx

    # Charges: all 1 except last vertex flipped if needed for odd total
    charges = [1] * n
    if n % 2 == 0:
        charges[-1] = 0  # total = n-1 (odd)

    axioms = []

    for v in range(n):
        e_left = (v, (v + 1) % n)
        e_right = ((v - 1) % n, v)

        xa = var_x.get(e_left)
        xb = var_x.get(e_right)

        if xa is not None and xb is not None:
            c = charges[v]
            # x_a + x_b - 2*x_a*x_b - c = 0
            axioms.append([
                (-float(c), frozenset()),
                (1.0, frozenset([xa])),
                (1.0, frozenset([xb])),
                (-2.0, frozenset([xa, xb])),
            ])

    # Boolean constraints: x_e * (1 - x_e) = 0
    # In multilinear representation this is x - x = 0 (trivial),
    # but we add x*(1-x) as an explicit axiom to help the IPS solver.
    for e, v_idx in var_x.items():
        # x_e - x_e^2 = 0 → in multilinear: 0 = 0 (trivially)
        # Instead add the useful form: x_e*(1-x_e) expanded
        # = x_e - x_e^2, but x_e^2 = x_e in frozenset representation
        # So this axiom would be 0. Skip it - booleanity is implicit.
        pass

    return axioms, num_vars, var_x


# --- k-XOR builder ---

def kxor_axioms(n, k=3, num_clauses=None, seed=42):
    """Build unsatisfiable k-XOR system via planted contradiction.

    Generates m ≈ 1.5*n random k-XOR clauses consistent with a planted
    assignment, then flips the last clause to create a contradiction.

    Each XOR constraint  x_{v1} ⊕ ... ⊕ x_{vk} = b  is encoded as a
    polynomial using elementary symmetric polynomials:

        sum_{j=1}^{k} (-2)^{j-1} * e_j(x_{vars}) - b = 0

    Returns (axioms, n, var_x) with no auxiliary variables.
    """
    rng = random.Random(seed)

    if num_clauses is None:
        num_clauses = max(k + 1, int(1.5 * n))

    # Variable map: x_0 .. x_{n-1}
    var_x = {i: i for i in range(n)}
    num_vars = n

    # Planted assignment
    a = [rng.randint(0, 1) for _ in range(n)]

    axioms = []
    for c_idx in range(num_clauses):
        # Pick k distinct variables
        vars_chosen = sorted(rng.sample(range(n), k))

        # Target parity under planted assignment
        b = 0
        for v in vars_chosen:
            b ^= a[v]

        # Flip last clause to make system unsatisfiable
        if c_idx == num_clauses - 1:
            b ^= 1

        # Encode as polynomial: sum_{j=1}^{k} (-2)^{j-1} * e_j(vars) - b = 0
        terms = [(-float(b), frozenset())]
        for j in range(1, k + 1):
            coeff = (-2.0) ** (j - 1)
            for subset in combinations(vars_chosen, j):
                terms.append((coeff, frozenset(subset)))

        axioms.append(terms)

    return axioms, num_vars, var_x


# --- Subset-Sum DP builder ---

def subset_sum_axioms(n_items, max_weight=4, seed=42):
    """Build unsatisfiable Subset-Sum instance with DP table as auxiliary vars.

    Weights are random EVEN numbers in [2, max_weight], target = 1 (odd),
    so no subset can reach the target.

    Variables:
      - var_x[i]: item selection (i = 0..n_items-1)
      - var_c[(i,s)]: DP cell "can items 0..i-1 reach sum s?"

    Axioms encode the DP recurrence plus a contradiction at c_{n, target}.

    Returns (axioms, num_vars, var_x, var_c).
    """
    rng = random.Random(seed)

    # Even weights → every reachable sum is even → target 1 is impossible
    weights = [2 * rng.randint(1, max_weight // 2) for _ in range(n_items)]
    target = 1
    capacity = sum(weights)

    # Variable indices
    idx = 0
    var_x = {}
    for i in range(n_items):
        var_x[i] = idx
        idx += 1

    var_c = {}
    for i in range(n_items + 1):
        for s in range(capacity + 1):
            var_c[(i, s)] = idx
            idx += 1
    num_vars = idx

    axioms = []

    # Base cases: c_{0,0} = 1, c_{0,s} = 0 for s > 0
    # c_{0,0} - 1 = 0
    axioms.append([
        (-1.0, frozenset()),
        (1.0, frozenset([var_c[(0, 0)]])),
    ])
    for s in range(1, capacity + 1):
        # c_{0,s} = 0
        axioms.append([
            (1.0, frozenset([var_c[(0, s)]])),
        ])

    # DP recurrence: c_{i,s} = (1 - x_i) * c_{i-1,s} + x_i * c_{i-1, s-w_i}
    # Rewrite: c_{i,s} - c_{i-1,s} + x_i * c_{i-1,s} - x_i * c_{i-1,s-w_i} = 0
    for i in range(1, n_items + 1):
        w_i = weights[i - 1]
        xi = var_x[i - 1]
        for s in range(capacity + 1):
            c_is = var_c[(i, s)]
            c_prev_s = var_c[(i - 1, s)]

            terms = [
                (1.0, frozenset([c_is])),       # + c_{i,s}
                (-1.0, frozenset([c_prev_s])),   # - c_{i-1,s}
                (1.0, frozenset([xi, c_prev_s])),  # + x_i * c_{i-1,s}
            ]

            if s - w_i >= 0:
                c_prev_sw = var_c[(i - 1, s - w_i)]
                terms.append((-1.0, frozenset([xi, c_prev_sw])))  # - x_i * c_{i-1,s-w_i}

            axioms.append(terms)

    # Contradiction: c_{n, target} = 1 (but no subset reaches odd target)
    axioms.append([
        (-1.0, frozenset()),
        (1.0, frozenset([var_c[(n_items, target)]])),
    ])

    return axioms, num_vars, var_x, var_c


# --- Job builders ---

def build_job(system_type, n):
    """Build axioms and var_maps for a given system type and parameter n."""
    if system_type == "php":
        axioms, num_vars, var_x = php_axioms(n)
        return axioms, num_vars, {"var_x": var_x}

    elif system_type == "phpe":
        axioms, num_vars, var_x, var_y = phpe_axioms(n)
        return axioms, num_vars, {"var_x": var_x, "var_y": var_y}

    elif system_type == "phpc":
        axioms, num_vars, var_x, var_s = phpc_axioms(n)
        return axioms, num_vars, {"var_x": var_x, "var_s": var_s}

    elif system_type == "tseitin":
        axioms, num_vars, var_x = tseitin_axioms(n)
        return axioms, num_vars, {"var_x": var_x}

    elif system_type == "kxor":
        axioms, num_vars, var_x = kxor_axioms(n)
        return axioms, num_vars, {"var_x": var_x}

    elif system_type == "subset_sum":
        axioms, num_vars, var_x, var_c = subset_sum_axioms(n)
        return axioms, num_vars, {"var_x": var_x, "var_c": var_c}

    else:
        raise ValueError(f"Unknown system type: {system_type}")


# --- Main runner ---

class DiscoveryRunner:
    """Process discovery jobs from a queue, one per tick."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.engine = DiscoveryEngine(self.config)
        self.state = load_state()
        self.stop_requested = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\n>>> Stop requested. Finishing current job...")
        self.stop_requested = True

    def run(self):
        """Process all pending jobs."""
        os.makedirs(RESULTS_DIR, exist_ok=True)

        completed = set(self.state.get("completed", []))
        pending = [(name, stype, n) for name, stype, n in JOB_QUEUE
                    if name not in completed]

        if not pending:
            print("All jobs completed!")
            return

        print("=== Selector Discovery Runner ===")
        print(f"Jobs: {len(pending)} pending, {len(completed)} completed")
        print(f"Results dir: {RESULTS_DIR}")
        print()

        for job_name, system_type, n in pending:
            if self.stop_requested:
                print("Stopped by user.")
                break

            print(f"--- Job: {job_name} ---")
            self.state["current"] = job_name
            self.state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(self.state)

            try:
                axioms, num_vars, var_maps = build_job(system_type, n)
                result = self.engine.run(job_name, axioms, num_vars, var_maps)

                # Save result JSON
                safe_name = job_name.replace("(", "_").replace(")", "")
                result_path = os.path.join(RESULTS_DIR, f"{safe_name}.json")
                result.save(result_path)
                print(f"  Saved: {result_path}")

                # Update state
                self.state["completed"].append(job_name)
                self.state["current"] = None
                save_state(self.state)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                self.state["current"] = None
                save_state(self.state)

        print("\n=== Done ===")
        completed = self.state.get("completed", [])
        print(f"Completed: {len(completed)}/{len(JOB_QUEUE)}")


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(description="Selector Discovery Runner")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as background daemon")
    parser.add_argument("--reset", action="store_true",
                        help="Reset state and start from scratch")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("State reset.")

    if args.daemon:
        pid = os.fork()
        if pid > 0:
            print(f"Discovery runner started as daemon (PID {pid})")
            sys.exit(0)
        os.setsid()
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "discovery.log")
        sys.stdout = open(log_path, "a")
        sys.stderr = sys.stdout

    runner = DiscoveryRunner()
    runner.run()


if __name__ == "__main__":
    main()
