import os
"""Hori-Vafa polynomials of the 18 smooth toric Fano threefolds, built from
toric descriptions; each is checked against the paper's polynomial by comparing
period sequences (constant terms of W^n), then its critical data is computed."""
import sys, time
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms, periods_mod, x, y, z
from toric import crit_data, vertices, is_smooth_fano, CHI


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


a, b, c = (1, 0, 0), (0, 1, 0), (0, 0, 1)
d = (-1, -1, -1)
P2 = [(1, 0, 0), (0, 1, 0), (-1, -1, 0)]
P1 = [(0, 0, 1), (0, 0, -1)]
F1 = [(1, 0, 0), (0, 1, 0), (-1, -1, 0), (1, 1, 0)]
S7 = F1 + [(-1, 0, 0)]
S6 = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0)]
e = add(a, b, c)

FANS = {
    "P3": [a, b, c, d],
    "MM_2_33": [a, b, c, d, add(a, b)],
    "MM_2_34": P2 + P1,
    "MM_2_35": [a, b, c, d, e],
    "MM_2_36": [(1, 0, 0), (0, 1, 0), (-1, -1, 2), (0, 0, 1), (0, 0, -1)],
    "MM_3_25": [a, b, c, d, add(a, b), add(c, d)],
    "MM_3_26": [a, b, c, d, e, add(a, d)],
    "MM_3_27": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "MM_3_28": F1 + P1,
    "MM_3_29": [a, b, c, d, e, add(e, a)],
    "MM_3_30": [a, b, c, d, e, add(a, b)],
    "MM_3_31": [(1, 0, 0), (-1, 0, 1), (0, 1, 0), (0, -1, 1), (0, 0, 1), (0, 0, -1)],
    "MM_4_9": [a, b, c, d, add(a, b), add(c, d), add(a, b, c)],
    "MM_4_10": S7 + P1,
    "MM_4_11": F1 + P1 + [(1, 1, 1)],
    "MM_4_12": [a, b, c, d, add(a, b), add(a, b, c), add(a, b, d)],
    "MM_5_2": [a, b, c, d, add(a, b), add(c, d), add(a, b, c), add(a, b, d)],
    "MM_5_3": S6 + P1,
}


def hv(rays):
    return sum(x**r[0] * y**r[1] * z**r[2] for r in rays)


if __name__ == "__main__":
    models = all_models()
    P, N = 2147483629, 14
    for lab in (sys.argv[1:] or list(FANS)):
        rays = FANS[lab]
        t0 = time.time()
        W = models[lab]
        terms = laurent_terms(W)
        const = dict(terms).get((0, 0, 0), 0)
        H = hv(rays) + const
        verts, ineqs = vertices(list(set(rays)))
        smooth = len(verts) == len(rays) and is_smooth_fano(verts, ineqs)
        nfacets = len(ineqs)
        cW = periods_mod(terms, N, P)
        cH = periods_mod(laurent_terms(H), N, P)
        same = cW == cH
        line = (f"{lab:8s} chi={CHI[lab]:2d} smoothFano={smooth} facets={nfacets} "
                f"periods_match={same}")
        if not same:
            line += f"  W:{cW[:7]} HV:{cH[:7]}"
        print(line, flush=True)
        if same and smooth:
            mu, nd, vals = crit_data(H)
            print(f"    HV: mu_tor={mu} distinct_crit_values={nd} [{time.time()-t0:.1f}s]", flush=True)
