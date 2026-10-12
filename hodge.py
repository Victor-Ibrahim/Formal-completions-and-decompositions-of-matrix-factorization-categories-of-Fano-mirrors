# Independent computation of Euler characteristics e(X) and h^{1,2} = (4 - e)/2 for the ten Fano threefolds,
# plus degree H^3 and index (c1 = r H) as sanity checks.
import itertools, sympy as sp
from fractions import Fraction as Fr
eps = sp.Symbol('eps'); h = sp.Symbol('h')

def ci_weighted(a, d):
    """smooth CI of degrees d in P(a) (avoiding singular points): returns (e, H^3, index)"""
    c = sp.prod([(1+ai*h) for ai in a]) / sp.prod([(1+dj*h) for dj in d])
    s = sp.series(c, h, 0, 4).removeO()
    c3 = s.coeff(h,3); H3 = Fr(int(sp.prod(d)), int(sp.prod(a)))
    index = sum(a) - sum(d)
    return Fr(int(sp.nsimplify(c3*sp.prod(d)).p), int(sp.nsimplify(c3*sp.prod(d)).q)) / int(sp.prod(a)) if False else sp.Rational(c3)*sp.Rational(H3.numerator,H3.denominator), H3, index

def grass_zero_locus(k, n, bundles, ts=None):
    """X = zero locus in Gr(k,n) of a section of E = sum of bundles; each bundle is a function
    (Uw, Qw) -> list of weights. Returns (e(X), H^3, c1 coefficient)."""
    import random
    ts = ts or [Fr(random.randint(-10**6,10**6), random.randint(1,10**3)) for _ in range(n)]
    dimG = k*(n-k)
    eX = Fr(0); H3 = Fr(0); c1s = Fr(0)
    for I in itertools.combinations(range(n), k):
        Uw = [ts[i] for i in I]; Qw = [ts[j] for j in range(n) if j not in I]
        Tw = [q - u for u in Uw for q in Qw]
        Ew = [w for b in bundles for w in b(Uw,Qw)]
        eT = 1
        for w in Tw: eT *= w
        ctop = 1
        for w in Ew: ctop *= w
        # c(T_G)/c(E) up to degree 3 via power sums -> use polynomial in eps with Fractions
        def cpoly(ws, deg):
            p = [Fr(1)] + [Fr(0)]*deg
            for w in ws:
                for i in range(deg,0,-1): p[i] += p[i-1]*w
            return p
        cT = cpoly(Tw,3); cE = cpoly(Ew,3)
        inv = [Fr(1),Fr(0),Fr(0),Fr(0)]
        for i in range(1,4): inv[i] = -sum(cE[j]*inv[i-j] for j in range(1,i+1))
        q = [sum(cT[j]*inv[i-j] for j in range(i+1)) for i in range(4)]
        Hw = -sum(Uw)        # c1(O(1)) = c1(U^*)
        eX += ctop*q[3]/eT
        H3 += ctop*Hw**3/eT
        c1s += ctop*q[1]*Hw**2/eT
    return eX, H3, c1s/H3

O = lambda m: (lambda Uw,Qw: [-m*sum(Uw)])
def QdualO1(Uw,Qw): return [-q - sum(Uw) for q in Qw]
def L2Udual(Uw,Qw): return [-(Uw[i]+Uw[j]) for i in range(len(Uw)) for j in range(i+1,len(Uw))]

rows = []
rows.append(('B1  X6 in P(1,1,1,2,3)',)+ci_weighted([1,1,1,2,3],[6]))
rows.append(('B2  X4 in P(1,1,1,1,2)',)+ci_weighted([1,1,1,1,2],[4]))
rows.append(('B3  X3 in P^4',)+ci_weighted([1]*5,[3]))
rows.append(('B4  X(2,2) in P^5',)+ci_weighted([1]*6,[2,2]))
rows.append(('B5  Gr(2,5) cap H^3',)+grass_zero_locus(2,5,[O(1)]*3))
rows.append(('V6  X(2,3) in P^5',)+ci_weighted([1]*6,[2,3]))
rows.append(('V10 Gr(2,5) cap H^2 cap Q',)+grass_zero_locus(2,5,[O(1),O(1),O(2)]))
rows.append(('V14 Gr(2,6) cap H^5',)+grass_zero_locus(2,6,[O(1)]*5))
rows.append(('V18 Z(Q^*(1)) in Gr(2,7) cap H^2',)+grass_zero_locus(2,7,[QdualO1,O(1),O(1)]))
rows.append(('V22 Z((L2 U^*)^3) in Gr(3,7)',)+grass_zero_locus(3,7,[L2Udual]*3))
print(f"{'X':36s} {'e(X)':>6s} {'H^3':>5s} {'index':>5s} {'h12=(4-e)/2':>12s} {'k=h12+1':>8s}")
for name,e,H3,ind in rows:
    e = sp.Rational(str(e)); h12 = (4-e)/2
    print(f"{name:36s} {str(e):>6s} {str(H3):>5s} {str(ind):>5s} {str(h12):>12s} {str(h12+1):>8s}")
