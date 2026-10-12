# Fano-side quantum periods via abelian/non-abelian + quantum Lefschetz, using a univariate
# epsilon-specialisation x_i = eps*a_i to extract the H^0 component.
from fractions import Fraction as Fr
from math import factorial, comb
import itertools, sympy as sp
def mul(p,q,D):
    r=[Fr(0)]*(D+1)
    for i,a in enumerate(p):
        if a:
            for j in range(D+1-i):
                if q[j]: r[i+j]+=a*q[j]
    return r
def lin(c, slope, D):  # c + slope*eps
    p=[Fr(0)]*(D+1); p[0]=Fr(c)
    if D>=1: p[1]=Fr(slope)
    return p
def inv(c, slope, D):  # 1/(c+slope*eps)
    return [Fr(1,1)/c * (Fr(-slope)/c)**j for j in range(D+1)]
def grass_I0(k, n, bundles, index, nterms, a=None):
    """bundles: list of (coeff vector over x_1..x_k, multiplicity, sign) sign=+1 numerator (convex summand)"""
    a = a or [Fr(p) for p in (3,7,13,29,53,101,211)[:k]]
    D = k*(k-1)//2
    Delta = Fr(1)
    for i in range(k):
        for j in range(i+1,k): Delta *= (a[i]-a[j])
    I=[Fr(0)]*(nterms+1)
    for d in range(0, nterms//index+1):
        tot=[Fr(0)]*(D+1)
        for ds in itertools.product(range(d+1), repeat=k):
            if sum(ds)!=d: continue
            p=[Fr(0)]*(D+1); p[0]=Fr((-1)**((k-1)*d))
            for i in range(k):
                for j in range(i+1,k): p=mul(p, lin(ds[i]-ds[j], a[i]-a[j], D), D)
            for i in range(k):
                for m in range(1,ds[i]+1):
                    q=inv(m, a[i], D)
                    for _ in range(n): p=mul(p,q,D)
            for cv,mult in bundles:
                deg = sum(c*e for c,e in zip(cv,ds)); sl = sum(c*e for c,e in zip(cv,a))
                for m in range(1,deg+1):
                    q=lin(m, sl, D)
                    for _ in range(mult): p=mul(p,q,D)
            tot=[u+v for u,v in zip(tot,p)]
        I[index*d] = tot[D]/Delta
    return I
def regularized(I, index):
    t=sp.Symbol('t'); c = I[1] if index==1 else 0
    ser=sp.expand(sp.series(sp.exp(-sp.Rational(c.numerator,c.denominator)*t)*sum(sp.Rational(v.numerator,v.denominator)*t**i for i,v in enumerate(I)),t,0,len(I)).removeO())
    return [factorial(i)*ser.coeff(t,i) for i in range(len(I))], c
NT = 7
def e(k,i): v=[0]*k; v[i]=1; return v
# re-derive the earlier ones (k=2,3) with this method as a cross-check
print("B5 :", regularized(grass_I0(2,5,[([1,1],3)],2,NT),2))
print("V10:", regularized(grass_I0(2,5,[([1,1],2),([2,2],1)],1,NT),1))
print("V14:", regularized(grass_I0(2,6,[([1,1],5)],1,NT),1))
print("V22:", regularized(grass_I0(3,7,[([1,1,0],3),([1,0,1],3),([0,1,1],3)],1,NT),1))
# V18 = zero locus in Gr(5,7) of Lambda^4 S^* + O(1)^2
L4 = [([1 if j!=i else 0 for j in range(5)],1) for i in range(5)]
print("V18:", regularized(grass_I0(5,7,L4+[([1]*5,2)],1,NT),1))

# Comparison with the Laurent polynomial in the paper, equation (21).
G18,c18 = regularized(grass_I0(5,7,L4+[([1]*5,2)],1,NT),1)
exec(open('periods.py').read().split('# index-2: B_d')[0].replace('N = 7','N = 7'))
fV18 = (x+y+z)*(x+y+z+x*y+x*z+y*z+x*y*z)/(x*y*z)
ct = const_terms(fV18, nterms=NT, shift=3)
print("V18 Laurent (21) minus 3:", ct, " matches the Fano side:", [int(u) for u in G18]==ct)
