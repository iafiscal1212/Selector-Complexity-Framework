from itertools import combinations

def php_axioms(n):
    pigeons = list(range(1, n + 2))
    holes = list(range(1, n + 1))
    var_x = {}
    idx = 0
    for p in pigeons:
        for h in holes:
            var_x[(p, h)] = idx
            idx += 1
    num_vars = idx
    axioms = []
    for p in pigeons:
        terms = []
        hvars = [var_x[(p, h)] for h in holes]
        for k in range(len(hvars) + 1):
            sign = (-1.0) ** k
            for subset in combinations(hvars, k):
                terms.append((sign, frozenset(subset)))
        axioms.append(terms)
    for h in holes:
        for i, p in enumerate(pigeons):
            for p2 in pigeons[i + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p2, h)]]))])
    for p in pigeons:
        for j, h in enumerate(holes):
            for h2 in holes[j + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p, h2)]]))])
    return axioms, num_vars, var_x

def phpe_axioms(n):
    pigeons = list(range(1, n + 2))
    holes = list(range(1, n + 1))
    var_x, var_y = {}, {}
    idx = 0
    for p in pigeons:
        for h in holes:
            var_x[(p, h)] = idx
            idx += 1
    for i, p in enumerate(pigeons):
        for p2 in pigeons[i + 1:]:
            var_y[(p, p2)] = idx
            idx += 1
    num_vars = idx
    axioms = []
    for p in pigeons:
        terms = []
        hvars = [var_x[(p, h)] for h in holes]
        for k in range(len(hvars) + 1):
            sign = (-1.0) ** k
            for subset in combinations(hvars, k):
                terms.append((sign, frozenset(subset)))
        axioms.append(terms)
    for h in holes:
        for i, p in enumerate(pigeons):
            for p2 in pigeons[i + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p2, h)]]))])
    for p in pigeons:
        for j, h in enumerate(holes):
            for h2 in holes[j + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p, h2)]]))])
    for i_p, p in enumerate(pigeons):
        for p2 in pigeons[i_p + 1:]:
            y_idx = var_y[(p, p2)]
            for h in holes:
                for h2 in holes:
                    if h == h2: continue
                    x1, x2 = var_x[(p, h)], var_x[(p2, h2)]
                    if h < h2:
                        axioms.append([(1.0, frozenset([x1, x2])), (-1.0, frozenset([x1, x2,
y_idx]))])
                    else:
                        axioms.append([(1.0, frozenset([x1, x2, y_idx]))])
    for i_p, p in enumerate(pigeons):
        for j_p, p2 in enumerate(pigeons[i_p + 1:], i_p + 1):
            for p3 in pigeons[j_p + 1:]:
                y12, y23, y13 = var_y[(p, p2)], var_y[(p2, p3)], var_y[(p, p3)]
                axioms.append([(1.0, frozenset([y12, y23])), (-1.0, frozenset([y12, y23,
y13]))])
                axioms.append([(1.0, frozenset([y13])), (-1.0, frozenset([y12, y13])),
                                (-1.0, frozenset([y23, y13])), (1.0, frozenset([y12, y23,
y13]))])
    return axioms, num_vars, var_x, var_y

def phpc_axioms(n):
    pigeons = list(range(1, n + 2))
    holes = list(range(1, n + 1))
    var_x, var_s = {}, {}
    idx = 0
    for p in pigeons:
        for h in holes:
            var_x[(p, h)] = idx
            idx += 1
    for p in pigeons:
        for q in pigeons:
            if p != q:
                var_s[(p, q)] = idx
                idx += 1
    num_vars = idx
    axioms = []
    for p in pigeons:
        terms = []
        hvars = [var_x[(p, h)] for h in holes]
        for k in range(len(hvars) + 1):
            sign = (-1.0) ** k
            for subset in combinations(hvars, k):
                terms.append((sign, frozenset(subset)))
        axioms.append(terms)
    for h in holes:
        for i, p in enumerate(pigeons):
            for p2 in pigeons[i + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p2, h)]]))])
    for p in pigeons:
        for j, h in enumerate(holes):
            for h2 in holes[j + 1:]:
                axioms.append([(1.0, frozenset([var_x[(p, h)], var_x[(p, h2)]]))])
    for p in pigeons:
        terms = []
        svars = [var_s[(p, q)] for q in pigeons if q != p]
        for k in range(len(svars) + 1):
            sign = (-1.0) ** k
            for subset in combinations(svars, k):
                terms.append((sign, frozenset(subset)))
        axioms.append(terms)
    for p in pigeons:
        others = [q for q in pigeons if q != p]
        for i, q in enumerate(others):
            for q2 in others[i + 1:]:
                axioms.append([(1.0, frozenset([var_s[(p, q)], var_s[(p, q2)]]))])
    for q in pigeons:
        terms = []
        svars = [var_s[(p, q)] for p in pigeons if p != q]
        for k in range(len(svars) + 1):
            sign = (-1.0) ** k
            for subset in combinations(svars, k):
                terms.append((sign, frozenset(subset)))
        axioms.append(terms)
    for q in pigeons:
        others = [p for p in pigeons if p != q]
        for i, p in enumerate(others):
            for p2 in others[i + 1:]:
                axioms.append([(1.0, frozenset([var_s[(p, q)], var_s[(p2, q)]]))])
    def succ_hole(h):
        return (h % n) + 1
    for p in pigeons:
        for q in pigeons:
            if p == q: continue
            s_idx = var_s[(p, q)]
            for h in holes:
                for h2 in holes:
                    if h2 == succ_hole(h): continue
                    axioms.append([(1.0, frozenset([s_idx, var_x[(p, h)], var_x[(q, h2)]]))])
    return axioms, num_vars, var_x, var_s
