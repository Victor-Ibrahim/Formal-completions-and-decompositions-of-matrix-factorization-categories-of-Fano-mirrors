import os
"""Compare torus critical values of each Laurent polynomial with the singular
values of the Picard-Fuchs recurrence of its classical period.

Period c_n = constant term of W^n (mod p).  The minimal-order linear recurrence
sum_k P_k(n) c_{n-k} = 0 has characteristic polynomial chi(l) = sum_k lc(P_k) l^(r-k);
its nonzero roots are the finite singular values of the period (singular fibres
of any compactification).  Torus critical values: elimination of the
logarithmic critical ideal plus W - s, saturated by xyz, over GF(p).
"""
import ast, sys, json, time, itertools
from fractions import Fraction
import numpy as np
import sympy as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mori_mukai_validation as MMV
import rank_one_fano_validation as R1

x, y, z = sp.symbols("x y z")


# ---------------------------------------------------------------- models
def rank_one_models():
    src = open(os.path.join(REPO, "rank_one_fano_validation.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    ns = dict(vars(R1))
    fn_to_label = {R1.CASES[k].__name__: k for k in R1.CASE_ORDER}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in fn_to_label:
            label = fn_to_label[node.name]
            local = dict(ns)
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    seg = ast.get_source_segment(src, stmt)
                    try:
                        exec(seg, local)
                    except Exception:
                        continue
                    if any(isinstance(t, ast.Name) and t.id == "potential" for t in stmt.targets):
                        out[label] = sp.sympify(local["potential"])
                        break
    missing = [k for k in R1.CASE_ORDER if k not in out]
    if missing:
        raise RuntimeError(f"rank-one potentials not found: {missing}")
    return {k: out[k] for k in R1.CASE_ORDER}


def all_models():
    models = rank_one_models()
    for k, c in MMV.CASES.items():
        models[k] = sp.sympify(c.potential)
    return models


def laurent_terms(W):
    """Return list of (exponent tuple, Fraction coefficient)."""
    num, den = sp.fraction(sp.together(sp.expand(W)))
    dpoly = sp.Poly(den, x, y, z)
    if len(dpoly.terms()) != 1:
        raise ValueError("denominator not a monomial")
    (dexp, dcoef), = dpoly.terms()
    terms = []
    for exp, coef in sp.Poly(sp.expand(num), x, y, z).terms():
        e = tuple(a - b for a, b in zip(exp, dexp))
        q = sp.Rational(coef) / sp.Rational(dcoef)
        terms.append((e, Fraction(int(q.p), int(q.q))))
    return terms


# ---------------------------------------------------------------- periods
def hull_inequalities(pts):
    """Facet inequalities a.v <= b of conv(pts) in Z^3 (brute force)."""
    P = np.array(pts, dtype=np.int64)
    ineqs = set()
    n = len(P)
    for i, j, k in itertools.combinations(range(n), 3):
        a = np.cross(P[j] - P[i], P[k] - P[i])
        if not a.any():
            continue
        g = np.gcd.reduce(np.abs(a))
        a = a // g
        vals = P @ a
        b = int(P[i] @ a)
        if (vals <= b).all():
            ineqs.add((tuple(int(t) for t in a), b))
        elif (vals >= b).all():
            ineqs.add((tuple(int(-t) for t in a), -b))
    return list(ineqs)


def periods_mod(terms, N, p):
    exps = np.array([e for e, _ in terms], dtype=np.int64)
    coefs = np.array([c.numerator * pow(c.denominator, -1, p) % p for _, c in terms], dtype=np.int64)
    ineqs = hull_inequalities([e for e, _ in terms] + [(0, 0, 0)])
    A = np.array([a for a, _ in ineqs], dtype=np.int64)
    B = np.array([b for _, b in ineqs], dtype=np.int64)
    span = int(np.abs(exps).max()) * N + 1
    base = 2 * span + 1

    def enc(m):
        return ((m[:, 0] + span) * base + (m[:, 1] + span)) * base + (m[:, 2] + span)

    def dec(k):
        c = k % base - span
        k = k // base
        b = k % base - span
        a = k // base - span
        return np.stack([a, b, c], axis=1)

    mons = np.zeros((1, 3), dtype=np.int64)
    vals = np.ones(1, dtype=np.int64)
    out = [1]
    zero_key = enc(np.zeros((1, 3), dtype=np.int64))[0]
    for n in range(1, N + 1):
        allm = (mons[None, :, :] + exps[:, None, :]).reshape(-1, 3)
        allv = ((vals[None, :] * coefs[:, None]) % p).reshape(-1)
        # prune: remaining steps must bring monomial back to 0: -m in (N-n) P
        rem = N - n
        ok = ((-allm) @ A.T <= rem * B[None, :]).all(axis=1)
        allm, allv = allm[ok], allv[ok]
        keys = enc(allm)
        order = np.argsort(keys, kind="stable")
        keys, allv = keys[order], allv[order]
        uniq, start = np.unique(keys, return_index=True)
        sums = np.add.reduceat(allv, start) % p
        nz = sums != 0
        mons, vals = dec(uniq[nz]), sums[nz]
        hit = np.where(uniq[nz] == zero_key)[0]
        out.append(int(vals[hit[0]]) if len(hit) else 0)
    return out


# ---------------------------------------------------------------- GF(p) linear algebra
def nullspace_mod(M, p):
    M = [list(r) for r in M]
    rows, cols = len(M), len(M[0])
    piv = []
    r = 0
    for c in range(cols):
        pr = next((i for i in range(r, rows) if M[i][c] % p), None)
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        inv = pow(M[r][c], -1, p)
        M[r] = [v * inv % p for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] % p:
                f = M[i][c]
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for fcol in free:
        v = [0] * cols
        v[fcol] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-M[i][fcol]) % p
        basis.append(v)
    return basis


def poly_trim(a, p):
    a = [c % p for c in a]
    while a and a[-1] == 0:
        a.pop()
    return a


def poly_divmod(a, b, p):
    a = poly_trim(a, p); b = poly_trim(b, p)
    q = [0] * max(len(a) - len(b) + 1, 1)
    inv = pow(b[-1], -1, p)
    while len(a) >= len(b) and a:
        f = a[-1] * inv % p
        s = len(a) - len(b)
        q[s] = f
        for i, c in enumerate(b):
            a[s + i] = (a[s + i] - f * c) % p
        a = poly_trim(a, p)
    return poly_trim(q, p), a


def poly_gcd(a, b, p):
    a = poly_trim(a, p); b = poly_trim(b, p)
    while b:
        _, r = poly_divmod(a, b, p)
        a, b = b, r
    if a:
        inv = pow(a[-1], -1, p)
        a = [c * inv % p for c in a]
    return a


def poly_deriv(a, p):
    return poly_trim([i * a[i] % p for i in range(1, len(a))], p)


def squarefree_nonzero(a, p):
    """Squarefree part with the factor l^k removed (coefficients low->high)."""
    a = poly_trim(a, p)
    while a and a[0] == 0:
        a = a[1:]
    if len(a) <= 1:
        return [1]
    g = poly_gcd(a, poly_deriv(a, p), p)
    q, _ = poly_divmod(a, g, p)
    inv = pow(q[-1], -1, p)
    return [c * inv % p for c in q]


def gap(c):
    from math import gcd
    g = 0
    for n, v in enumerate(c):
        if n and v:
            g = gcd(g, n)
    return g or 1


def guess_chi(c, p, **kw):
    """chi in the variable l, accounting for lacunary periods c_n (n = 0 mod g)."""
    g = gap(c)
    a = c[::g]
    rec = guess_recurrence(a, p, **kw)
    if rec is None:
        return None
    mu = rec["chi"]                      # polynomial in mu = l^g
    chi = [0] * ((len(mu) - 1) * g + 1)
    for i, cf in enumerate(mu):
        chi[i * g] = cf
    rec = dict(rec, gap=g, chi=chi, terms=len(a))
    return rec


def guess_recurrence(c, p, rmax=12, dmax=24, margin=10):
    N = len(c) - 1
    for r in range(1, rmax + 1):
        for d in range(0, dmax + 1):
            unknowns = (r + 1) * (d + 1)
            neq = N - r + 1
            if neq < unknowns + margin:
                break
            M = []
            for n in range(r, N + 1):
                row = []
                for k in range(r + 1):
                    for j in range(d + 1):
                        row.append(pow(n, j, p) * c[n - k] % p)
                M.append(row)
            ns = nullspace_mod(M, p)
            if ns:
                chis = []
                for v in ns:
                    P = [v[k * (d + 1):(k + 1) * (d + 1)] for k in range(r + 1)]
                    D = max((j for k in range(r + 1) for j in range(d + 1) if P[k][j] % p), default=0)
                    chi = [0] * (r + 1)  # coefficient of l^(r-k) is lc_D(P_k)
                    for k in range(r + 1):
                        chi[r - k] = P[k][D] % p
                    chis.append(chi)
                g = chis[0]
                for ch in chis[1:]:
                    g = poly_gcd(g, ch, p)
                return {"order": r, "degree": d, "nullity": len(ns), "chi": g}
    return None


# ---------------------------------------------------------------- torus critical values
def torus_disc_mod(W, p, timeout_note=""):
    s, u = sp.symbols("s u")
    num, den = sp.fraction(sp.together(sp.expand(W)))
    th = [sp.fraction(sp.together(sp.expand(v * sp.diff(W, v))))[0] for v in (x, y, z)]
    eqs = [sp.expand(e) for e in th] + [sp.expand(num - s * den), u * x * y * z - 1]
    G = sp.groebner(eqs, u, x, y, z, s, order="lex", modulus=p)
    elim = [g for g in G.exprs if g.free_symbols <= {s}]
    if not elim:
        return None
    poly = sp.Poly(elim[0], s, modulus=p)
    coeffs = [int(c) % p for c in reversed(poly.all_coeffs())]
    return coeffs


if __name__ == "__main__":
    labels = sys.argv[1:]
    models = all_models()
    print(len(models), "models")
    P = 2147483629
    for lab in labels:
        t0 = time.time()
        terms = laurent_terms(models[lab])
        c = periods_mod(terms, int(__import__("os").environ.get("NTERMS", "60")), P)
        rec = guess_chi(c, P)
        t1 = time.time()
        chi = squarefree_nonzero(rec["chi"], P) if rec else None
        D = torus_disc_mod(models[lab], P)
        t2 = time.time()
        Dsf = squarefree_nonzero(D, P) if D else None
        g = poly_gcd(chi, Dsf, P) if chi and Dsf else None
        print(lab, "c:", c[:8], "rec:", rec and (rec["gap"], rec["order"], rec["degree"], rec["nullity"]),
              "#PF:", chi and len(chi) - 1, "#torus:", Dsf and len(Dsf) - 1,
              "#common:", g and len(g) - 1, "0crit:", D and D[0] % P == 0,
              f"t={t1-t0:.1f}+{t2-t1:.1f}s")
