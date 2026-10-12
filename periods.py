# Period sequences: Fano side (quantum periods via (abelian/non-abelian) quantum Lefschetz)
# vs. constant terms of the Laurent polynomials used in the paper.
from fractions import Fraction as Fr
from math import factorial
import itertools, sympy as sp
N = 7   # number of terms

# ---------- truncated multivariate series in x1..xk up to total degree D ----------
class TS:
    def __init__(s, k, D, terms=None): s.k=k; s.D=D; s.t = terms or {}
    @staticmethod
    def const(k,D,c): return TS(k,D,{(0,)*k: Fr(c)})
    def __mul__(s,o):
        r = {}
        for e1,c1 in s.t.items():
            for e2,c2 in o.t.items():
                e = tuple(a+b for a,b in zip(e1,e2))
                if sum(e) <= s.D: r[e] = r.get(e,0) + c1*c2
        return TS(s.k,s.D,r)
    def __add__(s,o):
        r = dict(s.t)
        for e,c in o.t.items(): r[e] = r.get(e,0)+c
        return TS(s.k,s.D,r)
def lin(k,D,coeffs,c):   # c + sum coeffs_i x_i
    t = {(0,)*k: Fr(c)}
    for i,a in enumerate(coeffs):
        if a: e=[0]*k; e[i]=1; t[tuple(e)] = Fr(a)
    return TS(k,D,t)
def inv_lin(k,D,coeffs,c):   # 1/(c + L) = (1/c) sum (-L/c)^j
    L = lin(k,D,coeffs,0); r = TS.const(k,D,0); p = TS.const(k,D,1)
    for j in range(D+1):
        r = r + TS(k,D,{e:v*Fr(1,c)*Fr(-1,c)**j for e,v in p.t.items()})
        p = p*L
    return r
def poch(k,D,coeffs,deg, power=1):  # prod_{m=1}^{deg} (L + m)^power ; deg<0 -> inverse convention
    r = TS.const(k,D,1)
    if deg >= 0:
        for m in range(1,deg+1):
            for _ in range(power): r = r*lin(k,D,coeffs,m)
    else:
        for m in range(deg+1,1):   # prod_{m=deg+1}^{0} (L+m)^{-1}... (not needed here)
            raise NotImplementedError
    return r
def ipoch(k,D,coeffs,deg,power=1):
    r = TS.const(k,D,1)
    for m in range(1,deg+1):
        for _ in range(power): r = r*inv_lin(k,D,coeffs,m)
    return r

def grass_period(k, n, num_bundles, den_bundles, index, nterms=N):
    """H^0-part of the twisted I-function of Gr(k,n) at z=1.
    num_bundles/den_bundles: list of (coeff_vector, multiplicity) for Chern roots rho = sum c_i x_i (x_i roots of U^*),
    the root's degree is sum c_i d_i.  Returns list I_0 coefficients in powers of t (t^{index*d})."""
    D = k*(k-1)//2
    vand_mon = tuple(range(k-1,-1,-1))   # x1^{k-1} x2^{k-2} ... coefficient of Vandermonde = 1
    res = {}
    maxd = nterms//index + 1
    for d in range(0, maxd+1):
        tot = TS.const(k,D,0)
        for ds in itertools.product(range(d+1), repeat=k):
            if sum(ds)!=d: continue
            Nn = TS.const(k,D,(-1)**((k-1)*d))
            for i in range(k):
                for j in range(i+1,k):
                    c=[0]*k; c[i]=1; c[j]=-1
                    Nn = Nn*lin(k,D,c,ds[i]-ds[j])
            for i in range(k):
                e=[0]*k; e[i]=1
                Nn = Nn*ipoch(k,D,e,ds[i],n)
            for cv,mult in num_bundles:
                Nn = Nn*poch(k,D,cv,sum(a*b for a,b in zip(cv,ds)),mult)
            for cv,mult in den_bundles:
                Nn = Nn*ipoch(k,D,cv,sum(a*b for a,b in zip(cv,ds)),mult)
            tot = tot + Nn
        res[index*d] = tot.t.get(vand_mon, Fr(0))
    I = [res.get(i,Fr(0)) for i in range(nterms+1)]
    return I

def to_G(I, index):
    # G = e^{-c t} I where c = coefficient of t^1 (only for index 1)
    t = sp.Symbol('t')
    c = I[1] if index==1 else 0
    ser = sp.expand(sp.series(sp.exp(-sp.Rational(c.numerator,c.denominator)*t)*sum(sp.Rational(a.numerator,a.denominator)*t**i for i,a in enumerate(I)), t, 0, len(I)).removeO())
    G = [ser.coeff(t,i) for i in range(len(I))]
    return [factorial(i)*G[i] for i in range(len(I))], c

def ci_period(a, d, index, nterms=N):
    I = [Fr(0)]*(nterms+1)
    for m in range(nterms//index+1):
        if index*m>nterms: break
        num = 1
        for dj in d: num *= factorial(dj*m)
        den = 1
        for ai in a: den *= factorial(ai*m)
        I[index*m] = Fr(num,den)
    return I

# ---------- Laurent constant terms ----------
x,y,z = sp.symbols('x y z')
def laurent_dict(f):
    f = sp.expand(f); d = {}
    for term in sp.Add.make_args(f):
        c, m = term.as_coeff_Mul()
        pw = m.as_powers_dict()
        e = tuple(int(pw.get(v,0)) for v in (x,y,z))
        d[e] = d.get(e,0) + int(c)
    return d
def const_terms(f, nterms=N, shift=0):
    d = laurent_dict(sp.expand(f - shift))
    cur = {(0,0,0):1}; out=[1]
    for n in range(1,nterms+1):
        new = {}
        for e1,c1 in cur.items():
            for e2,c2 in d.items():
                e = (e1[0]+e2[0],e1[1]+e2[1],e1[2]+e2[2]); new[e] = new.get(e,0)+c1*c2
        cur = new; out.append(cur.get((0,0,0),0))
    return out

def show(name, fano, laurents):
    print(f"\n### {name}\n  Fano side  : {[str(v) for v in fano]}")
    for lab,f,sh in laurents:
        ct = const_terms(f, shift=sh)
        print(f"  {lab:28s}: {ct}   match={[int(a)==int(b) if sp.Rational(a).q==1 else False for a,b in zip(fano,ct)]==[True]*len(ct)}")

# index-2: B_d
I = ci_period([1,1,1,2,3],[6],2); G,_ = to_G(I,2)
show("B1", G, [("std (x+y+1)^6/(xy^2z)+z", (x+y+1)**6/(x*y**2*z)+z, 0)])
I = ci_period([1,1,1,1,2],[4],2); G,_ = to_G(I,2)
FB = x*(y+z+1)+1/(x*y*z)
show("B2", G, [("compact (x+y+1)^4/(xyz)+z", (x+y+1)**4/(x*y*z)+z, 0), ("source F^2/x", sp.expand(FB**2/x), 0)])
I = ci_period([1]*5,[3],2); G,_ = to_G(I,2)
show("B3", G, [("compact (x+y+1)^3/(xyz)+z", (x+y+1)**3/(x*y*z)+z, 0),
               ("source W_B3", x + x/(y*z) + y + z + 2/z + 2/y + y/(x*z) + 2/x + z/(x*y), 0)])
I = ci_period([1]*6,[2,2],2); G,_ = to_G(I,2)
show("B4", G, [("(x+1)^2(y+1)^2/(xyz)+z", (x+1)**2*(y+1)**2/(x*y*z)+z, 0)])
I = grass_period(2,5,[([1,1],3)],[],2); G,_ = to_G(I,2)
show("B5", G, [("x+y+z+1/x+1/y+1/z+1/xyz", x+y+z+1/x+1/y+1/z+1/(x*y*z), 0)])
# index-1: V_{4d+2}
I = ci_period([1]*6,[2,3],1); G,c = to_G(I,1)
show(f"V6 (shift c={c})", G, [("(x+1)^2(y+z+1)^3/(xyz) - 12", (x+1)**2*(y+z+1)**3/(x*y*z), 12)])
I = grass_period(2,5,[([1,1],2),([2,2],1)],[],1); G,c = to_G(I,1)
F10 = x*y*z**2+x*y*z+1; G10 = x*y*z**2+x*y*z+x*z+y*z+1
show(f"V10 (shift c={c})", G, [("compact q^2/(xyz) - 6", (1+x+y+z+x*y+x*z+y*z)**2/(x*y*z), 6),
                             ("source W_V10", sp.expand(F10**2*G10/(x**2*y**2*z**3))-6, 0)])
I = grass_period(2,6,[([1,1],5)],[],1); G,c = to_G(I,1)
A14 = y*z**2+y*z+1; B14 = x**2*y*z**2 + x*y**2*z**4 + 2*x*y**2*z**3 + x*y**2*z**2 + 2*x*y*z + x + y*z**2
fV14 = (x+y+z+1)**2/x + (x+y+z+1)*(y+z+1)*(z+1)**2/(x*y*z)
show(f"V14 (shift c={c})", G, [("compact f_V14 - 4", fV14, 4), ("source W_V14", sp.expand(A14*B14/(x*y**2*z**3))-4, 0)])
# V18: ver periods2.py (Gr(5,7), Lambda^4 S^* + O(1)^2)
I = grass_period(3,7,[([1,1,0],3),([1,0,1],3),([0,1,1],3)],[],1); G,c = to_G(I,1)
fV22 = x*y/z + x + x/z + y + y/z + z + 4 + 1/z + z/y + 1/y + z/x + 1/x + z/(x*y) + 1/(x*y)
show(f"V22 (shift c={c})", G, [("f_V22 - 4", fV22, 4)])
