import os
"""Toric Fano threefolds: is the paper's Laurent polynomial the Hori-Vafa one,
how many torus critical points (with multiplicity) and distinct critical values?"""
import sys, itertools, time
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms, hull_inequalities, x, y, z

TORIC = ["P3", "MM_2_33", "MM_2_34", "MM_2_35", "MM_2_36",
         "MM_3_25", "MM_3_26", "MM_3_27", "MM_3_28", "MM_3_29", "MM_3_30", "MM_3_31",
         "MM_4_9", "MM_4_10", "MM_4_11", "MM_4_12", "MM_5_2", "MM_5_3"]
CHI = {"P3": 4, **{k: 6 for k in TORIC if k.startswith("MM_2")},
       **{k: 8 for k in TORIC if k.startswith("MM_3")},
       **{k: 10 for k in TORIC if k.startswith("MM_4")},
       **{k: 12 for k in TORIC if k.startswith("MM_5")}}


def vertices(pts):
    P = np.array(pts)
    ineqs = hull_inequalities(pts)
    verts = []
    for v in pts:
        tight = [a for a, b in ineqs if np.dot(a, v) == b]
        if np.linalg.matrix_rank(np.array(tight)) == 3 if tight else False:
            verts.append(tuple(v))
    return verts, ineqs


def is_smooth_fano(verts, ineqs):
    """Every facet is a unimodular simplex whose vertices are the vertices of the polytope."""
    for a, b in ineqs:
        if b != 1:
            return False
        fv = [v for v in verts if np.dot(a, v) == b]
        if len(fv) != 3 or abs(round(np.linalg.det(np.array(fv, dtype=float)))) != 1:
            return False
    return True


def crit_data(W):
    s, u = sp.symbols("s u")
    num, den = sp.fraction(sp.together(sp.expand(W)))
    th = [sp.expand(sp.fraction(sp.together(sp.expand(v * sp.diff(W, v))))[0]) for v in (x, y, z)]
    # multiplicity count: dim of C[x,y,z,u]/(theta, u xyz - 1)
    G = sp.groebner(th + [u * x * y * z - 1], u, x, y, z, order="grevlex")
    if list(G.exprs) == [1]:
        mu = 0
    else:
        lead = [sp.Poly(g, u, x, y, z).monoms(order="grevlex")[0] for g in G.exprs]
        # count standard monomials (finite if zero-dimensional)
        mu, bound = 0, 30
        for m in itertools.product(range(bound), repeat=4):
            if not any(all(m[i] >= l[i] for i in range(4)) for l in lead):
                mu += 1
        if mu > 2000:
            mu = None
    Gl = sp.groebner(th + [sp.expand(num - s * den), u * x * y * z - 1], u, x, y, z, s, order="lex")
    elim = [g for g in Gl.exprs if g.free_symbols <= {s}]
    D = sp.Poly(elim[0], s) if elim else None
    vals = sp.factor_list(D.as_expr()) if D is not None else None
    ndist = sum(sp.degree(f, s) for f, _ in vals[1]) if vals else None
    return mu, ndist, vals


if __name__ == "__main__":
    models = all_models()
    for lab in (sys.argv[1:] or TORIC):
        W = models[lab]
        t0 = time.time()
        terms = laurent_terms(W)
        pts = [e for e, _ in terms if e != (0, 0, 0)]
        verts, ineqs = vertices(pts)
        nonvert = [e for e, _ in terms if e not in verts and e != (0, 0, 0)]
        coefs = {e: c for e, c in terms}
        hv = (set(pts) == set(verts) and all(coefs[v] == 1 for v in verts)
              and is_smooth_fano(verts, ineqs))
        const = coefs.get((0, 0, 0), 0)
        try:
            mu, nd, vals = crit_data(W)
        except Exception as ex:
            mu, nd, vals = f"ERR {ex}", None, None
        print(f"{lab:8s} chi={CHI[lab]:2d} terms={len(terms):2d} verts={len(verts):2d} "
              f"nonvertex={len(nonvert)} const={const} HoriVafa={hv} mu_tor={mu} "
              f"distinct_crit_values={nd} [{time.time()-t0:.1f}s]", flush=True)
        if vals:
            print("    crit-value polynomial factors:", [(sp.factor(f), m) for f, m in vals[1]], flush=True)
