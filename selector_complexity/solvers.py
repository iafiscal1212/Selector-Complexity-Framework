import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from itertools import combinations

def build_matrix(axioms, num_vars, d_max):
    all_monoms = []
    monom_to_idx = {}
    for d in range(d_max + 1):
        for combo in combinations(range(num_vars), d):
            m = frozenset(combo)
            monom_to_idx[m] = len(all_monoms)
            all_monoms.append(m)
    num_monoms = len(all_monoms)
    rows, cols, vals = [], [], []
    total_unknowns = 0
    for ax in axioms:
        deg_ax = max(len(m) for c, m in ax)
        deg_mult = max(0, d_max - deg_ax)
        for d in range(deg_mult + 1):
            for combo in combinations(range(num_vars), d):
                m_mult = frozenset(combo)
                col = total_unknowns
                total_unknowns += 1
                for coef_ax, m_ax in ax:
                    m_prod = m_mult | m_ax
                    if len(m_prod) <= d_max and m_prod in monom_to_idx:
                        rows.append(monom_to_idx[m_prod])
                        cols.append(col)
                        vals.append(coef_ax)
    if total_unknowns == 0:
        return sparse.csr_matrix((num_monoms, 0)), np.zeros(num_monoms), num_monoms, 0
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(num_monoms, total_unknowns))
    b = np.zeros(num_monoms)
    b[monom_to_idx[frozenset()]] = 1.0
    return A, b, num_monoms, total_unknowns

def find_certificate(axioms, num_vars, d_max, atol=1e-12):
    A, b, nm, nu = build_matrix(axioms, num_vars, d_max)
    if nu == 0:
        return None
    res = lsqr(A, b, atol=atol, btol=atol, iter_lim=10000)
    x = res[0]
    residual = np.linalg.norm(A @ x - b)
    if residual < 1e-6:
        size = int(np.sum(np.abs(x) > 1e-8))
        return {"degree": d_max, "size": size, "num_monoms": nm,
                "num_unknowns": nu, "residual": float(residual), "coefficients": x}
    return None
