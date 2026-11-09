import os
"""Exact certificate for the table in sec:exceptional_comparison.

For each polynomial W (paper's, with corrected rank-four label, and Hori-Vafa):
  * mu = dim of the torus Jacobian ring (grevlex Groebner basis, exact);
  * chi_L = charpoly of multiplication by L = 3x+5y+7z, exact over Q.
    If chi_L is squarefree of degree mu, L takes mu distinct values on the
    critical points, so there are mu distinct points, each of Milnor number 1:
    all critical points are nondegenerate.
  * chi_W = charpoly of multiplication by W.  When all points are Morse, the
    multiplicity of a root lambda of chi_W is the number of critical points
    in W^{-1}(lambda) (points per fibre), read off from sqf_list.
Also: periods over Z to order 14 (exact), constant term = 0, smooth Fano
polytope of each fan, anticanonical degrees with the shifted rank-four
labels, and negative controls.  Every check is an assert.
"""
import sys, itertools
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms, x, y, z
from fans import FANS, hv
from compare import PAPER_LABEL
from toric import vertices, is_smooth_fano, CHI
from degree import degree, MM_DEGREE
from jac import to_poly, GENS, u

s = sp.Symbol("s")


def periods_Z(W, N):
    """Constant terms of W^n, n <= N, exactly over Z (sympy, small N)."""
    terms = laurent_terms(W)
    cur = {(0, 0, 0): sp.Integer(1)}
    out = [1]
    for n in range(1, N + 1):
        nxt = {}
        for m, c in cur.items():
            for e, a in terms:
                k = (m[0] + e[0], m[1] + e[1], m[2] + e[2])
                nxt[k] = nxt.get(k, 0) + c * sp.Rational(a.numerator, a.denominator)
        cur = {k: v for k, v in nxt.items() if v != 0}
        out.append(cur.get((0, 0, 0), 0))
    return out


def jac_matrices(W):
    th = [to_poly(v * sp.diff(W, v)) for v in (x, y, z)]
    G = sp.groebner(th + [u * x * y * z - 1], *GENS, order="grevlex", domain=sp.QQ)
    lead = [sp.Poly(g, *GENS).monoms(order="grevlex")[0] for g in G.exprs]
    std = []
    for m in itertools.product(range(40), repeat=4):
        if not any(all(m[i] >= l[i] for i in range(4)) for l in lead):
            std.append(m)
            assert len(std) <= 400, "positive-dimensional critical locus"
    basis = [sp.Mul(*[g**e for g, e in zip(GENS, m)]) for m in std]
    index = {m: i for i, m in enumerate(std)}

    def mat(f):
        fp = to_poly(f)
        M = sp.zeros(len(basis), len(basis))
        for j, b in enumerate(basis):
            _, r = G.reduce(sp.expand(fp * b))
            for mon, cf in sp.Poly(r, *GENS, domain=sp.QQ).terms():
                M[index[mon], j] = cf
        return M
    return len(basis), mat(W), mat(3 * x + 5 * y + 7 * z)


def certify(W):
    mu, MW, ML = jac_matrices(W)
    chiL = sp.Poly(ML.charpoly(s).as_expr(), s)
    sqfL = sp.quo(chiL, sp.gcd(chiL, chiL.diff(s)))
    morse = sp.degree(sqfL, s) == mu
    chiW = sp.Poly(MW.charpoly(s).as_expr(), s)
    sizes = []
    for f, k in sp.sqf_list(chiW)[1]:
        sizes += [k] * sp.degree(f, s)          # each root of f: k points
    assert sum(sizes) == mu
    return mu, morse, sorted(sizes, reverse=True), sp.factor(chiW.as_expr())


if __name__ == "__main__":
    models = all_models()
    print("== fans and periods ==", flush=True)
    for fam, rays in FANS.items():
        W = models[PAPER_LABEL[fam]]
        const = dict(laurent_terms(W)).get((0, 0, 0), 0)
        assert const == 0, (fam, const)
        verts, ineqs = vertices(list(set(rays)))
        assert len(verts) == len(rays) and is_smooth_fano(verts, ineqs), fam
        assert len(ineqs) == CHI[fam], (fam, len(ineqs))
        cW, cH = periods_Z(W, 14), periods_Z(hv(rays), 14)
        assert cW == cH, (fam, cW[:8], cH[:8])
        dW = round(degree([e for e, _ in laurent_terms(W) if e != (0, 0, 0)]), 6)
        dF = round(degree(rays), 6)
        assert dW == dF == MM_DEGREE[fam], (fam, dW, dF, MM_DEGREE[fam])
        print(f"{fam:8s} listed as {PAPER_LABEL[fam]:8s} smooth-Fano fan, facets={len(ineqs)}, "
              f"periods equal over Z to order 14, degree {MM_DEGREE[fam]} (paper polytope = fan = MM)", flush=True)

    print("\n== negative controls ==", flush=True)
    for fam in ("MM_4_9", "MM_4_10", "MM_4_11", "MM_4_12"):
        W0 = models[fam]                      # unshifted label
        ok = periods_Z(W0, 14) == periods_Z(hv(FANS[fam]), 14)
        assert not ok, fam
        print(f"{fam}: unshifted polynomial vs fan -> periods differ (expected)", flush=True)
    assert periods_Z(hv(FANS["MM_3_29"]), 14) != periods_Z(models["MM_3_30"], 14)
    print("MM_3_29 fan vs MM_3_30 polynomial -> periods differ (expected)", flush=True)
    bad = [r for r in FANS["MM_2_33"] if r != (1, 1, 0)] + [(1, 1, 1)]
    v, i = vertices(list(set(bad)))
    assert periods_Z(hv(bad), 14) != periods_Z(models["MM_2_33"], 14)
    print("MM_2_33 fan with a+b replaced by (1,1,1) -> periods differ (expected)", flush=True)
    Wb = models["MM_2_33"] + x                # perturbed coefficient
    assert periods_Z(Wb, 6) != periods_Z(models["MM_2_33"], 6)
    print("MM_2_33 polynomial with one coefficient changed -> periods differ (expected)", flush=True)

    print("\n== CCGK polynomials replaced by Hori-Vafa in the paper ==", flush=True)
    OLD_CCGK = {
        "MM_3_31": (x**2/(y**2*z) + x + 2*x/y + y + z + 1/x, 4),
        "MM_4_10": (x + y + z + 1/y + z/x + 2/x + 1/(x*z), 6),
        "MM_5_2": (x + y + y/z + z + 1/y + y/x + 2/x + 1/(x*y), 8),
        "MM_5_3": (x + y + z + 1/y + 3/x + 2/(x*z) + 3/(x**2*z) + 1/(x**3*z**2), 2),
    }
    for fam, (Wold, mu_old) in OLD_CCGK.items():
        assert periods_Z(Wold, 14) == periods_Z(hv(FANS[fam]), 14), fam
        mu, morse, sizes, _ = certify(Wold)
        assert mu == mu_old and morse, (fam, mu, morse)
        print(f"{fam:8s} CCGK polynomial: periods equal to HV over Z to order 14; "
              f"mu={mu} < n={CHI[fam]}, points-per-fibre={sizes}", flush=True)

    print("\n== exact critical data ==", flush=True)
    rows = list(FANS) + ["Q", "B_5", "V_22"]
    for fam in rows:
        lab = PAPER_LABEL.get(fam, fam)
        mu, morse, sizes, chiW = certify(models[lab])
        line = f"{fam:8s} paper({lab}): mu={mu} morse={morse} points-per-fibre={sizes}"
        if fam in FANS:
            muH, morseH, sizesH, chiH = certify(hv(FANS[fam]))
            assert muH == CHI[fam] and morseH, (fam, muH, morseH)
            line += f" | HV: mu={muH} morse={morseH} points-per-fibre={sizesH}"
        print(line, flush=True)
    print("\nALL CHECKS PASSED", flush=True)
