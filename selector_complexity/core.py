"""Core definitions: PolynomialSystem, SelectorFamily."""

import numpy as np
from itertools import combinations


class PolynomialSystem:
    """A system of polynomial equations over boolean variables."""

    def __init__(self, name, num_vars, axioms):
        self.name = name
        self.num_vars = num_vars
        self.axioms = axioms

    def evaluate(self, assignment):
        values = []
        for ax in self.axioms:
            val = 0.0
            for coef, monom in ax:
                prod = coef
                for v in monom:
                    prod *= assignment.get(v, 0)
                val += prod
            values.append(val)
        return values

    def is_unsatisfiable(self, max_vars=15):
        if self.num_vars > max_vars:
            return None
        for bits in range(2 ** self.num_vars):
            assignment = {v: (bits >> v) & 1 for v in range(self.num_vars)}
            vals = self.evaluate(assignment)
            if all(abs(v) < 1e-10 for v in vals):
                return False
        return True

    def __repr__(self):
        return "PolynomialSystem({}, {} vars, {} axioms)".format(
            self.name, self.num_vars, len(self.axioms))


class SelectorFamily:
    """A family of selector polynomials."""

    def __init__(self, selectors, var_assignments_generator):
        self.selectors = selectors
        self.var_gen = var_assignments_generator

    def verify(self):
        labels = list(self.selectors.keys())
        partition_ok = exclusivity_ok = boolean_ok = total = 0
        for assignment in self.var_gen():
            total += 1
            vals = {}
            for label in labels:
                v = sum(c * np.prod([assignment.get(var, 0) for var in m])
                        for c, m in self.selectors[label])
                vals[label] = v
            if all(abs(v) < 1e-10 or abs(v - 1) < 1e-10 for v in vals.values()):
                boolean_ok += 1
            if abs(sum(vals.values()) - 1.0) < 1e-10:
                partition_ok += 1
            excl = all(abs(vals[l1] * vals[l2]) < 1e-10
                        for i, l1 in enumerate(labels)
                        for l2 in labels[i+1:])
            if excl:
                exclusivity_ok += 1
        return {"total": total, "partition": partition_ok,
                "exclusivity": exclusivity_ok, "boolean": boolean_ok,
                "all_pass": partition_ok == total == exclusivity_ok == boolean_ok}

    def size(self):
        return sum(len(p) for p in self.selectors.values())

    def max_degree(self):
        return max(max((len(m) for _, m in p), default=0)
                    for p in self.selectors.values())
