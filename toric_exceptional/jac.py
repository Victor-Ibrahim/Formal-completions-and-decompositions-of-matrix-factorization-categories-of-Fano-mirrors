import os
"""Torus Jacobian ring and multiplication by W.

mu_tor = dim C[x^+-,y^+-,z^+-]/(theta W); the eigenvalues of multiplication by W on
it are the critical values with multiplicity (only defined when the critical
locus is finite).  Number of distinct critical values = degree of the
squarefree part of its characteristic polynomial."""
import sys, time, itertools
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms, x, y, z

u = sp.Symbol("u")
GENS = (u, x, y, z)


def to_poly(expr):
    """Laurent expression -> polynomial in u,x,y,z with u = 1/(xyz)."""
    num, den = sp.fraction(sp.together(sp.expand(expr)))
    (dexp, dcoef), = sp.Poly(den, x, y, z).terms()
    mx = max(dexp)
    inv = u**mx * x**(mx - dexp[0]) * y**(mx - dexp[1]) * z**(mx - dexp[2])
    return sp.expand(num * inv / dcoef)


def jacobian_data(W):
    th = [to_poly(v * sp.diff(W, v)) for v in (x, y, z)]
    G = sp.groebner(th + [u * x * y * z - 1], *GENS, order="grevlex")
    if G.exprs == [1]:
        return 0, 0, sp.Integer(1)
    lead = [sp.Poly(g, *GENS).monoms(order="grevlex")[0] for g in G.exprs]
    std = []
    for m in itertools.product(range(40), repeat=4):
        if not any(all(m[i] >= l[i] for i in range(4)) for l in lead):
            std.append(m)
            if len(std) > 400:
                return None, None, None        # positive-dimensional
    basis = [sp.Mul(*[g**e for g, e in zip(GENS, m)]) for m in std]
    index = {m: i for i, m in enumerate(std)}
    Wp = to_poly(W)
    n = len(basis)
    M = sp.zeros(n, n)
    for j, b in enumerate(basis):
        _, r = G.reduce(sp.expand(Wp * b))
        for mon, cf in sp.Poly(r, *GENS).terms():
            M[index[mon], j] = cf
    s = sp.Symbol("s")
    cp = M.charpoly(s).as_expr()
    sqf = sp.quo(cp, sp.gcd(cp, sp.diff(cp, s)), s)
    return n, sp.degree(sqf, s), sp.factor(cp)


if __name__ == "__main__":
    from fans import FANS, hv
    from compare import PAPER_LABEL
    from toric import CHI
    models = all_models()
    fams = sys.argv[1:] or list(FANS)
    print("family   chi | paper-label  paper: mu #values | HV: mu #values", flush=True)
    for fam in fams:
        W = models[PAPER_LABEL[fam]]
        const = dict(laurent_terms(W)).get((0, 0, 0), 0)
        H = hv(FANS[fam]) + const
        t0 = time.time()
        hp = jacobian_data(H)
        pp = jacobian_data(W)
        print(f"{fam:8s} {CHI[fam]:3d} | {PAPER_LABEL[fam]:10s} {str(pp[0]):>6s} {str(pp[1]):>7s} | "
              f"{str(hp[0]):>5s} {str(hp[1]):>7s}   [{time.time()-t0:.1f}s]", flush=True)
        print(f"    HV charpoly: {hp[2]}", flush=True)
        print(f"    paper charpoly: {pp[2]}", flush=True)
