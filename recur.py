# Guess the linear recurrence (Picard-Fuchs) satisfied by the period sequence and read off the
# singular points: roots of the characteristic polynomial = critical values (eigenvalues of c1*).
import sympy as sp, sys
from fractions import Fraction as Fr
exec(open('periods.py').read().split('def show')[0].replace('N = 7','N = 7'))
def const_seq(f, n, shift):
    d = laurent_dict(sp.expand(f - shift)); cur={(0,0,0):1}; out=[1]
    for k in range(1,n+1):
        new={}
        for e1,c1 in cur.items():
            for e2,c2 in d.items():
                e=(e1[0]+e2[0],e1[1]+e2[1],e1[2]+e2[2]); new[e]=new.get(e,0)+c1*c2
        cur=new; out.append(cur.get((0,0,0),0))
    return out
def guess(a, r, s):
    n0 = r; rows=[]
    for n in range(n0, len(a)):
        rows.append([sp.Integer(n)**j * a[n-i] for i in range(r+1) for j in range(s+1)])
    M = sp.Matrix(rows); ns = M.nullspace()
    return ns
def analyse(name, f, shift, crit, N=34):
    a = const_seq(f, N, shift)
    for r in range(1,9):
        found=False
        for s in range(0,5):
            if (r+1)*(s+1) >= len(a)-r-4: break
            ns = guess(a, r, s)
            if ns:
                v = ns[0]; P = [[v[i*(s+1)+j] for j in range(s+1)] for i in range(r+1)]
                # characteristic polynomial: sum_i lead_i * X^{r-i}, lead_i = coefficient of n^s in P_i
                X = sp.Symbol('X')
                ch = sp.factor(sum(P[i][s]*X**(r-i) for i in range(r+1)))
                roots = sp.Poly(sp.sqf_part(sp.Poly(sum(P[i][s]*X**(r-i) for i in range(r+1)), X)), X).nroots(maxsteps=500)
                print(f"{name}: recurrence order {r}, degree {s}; char. poly = {ch}")
                print(f"   nonzero roots ~ {[complex(round(sp.re(z),6), round(sp.im(z),6)) for z in roots if abs(z)>1e-9]}")
                print(f"   expected (shifted critical values): {[complex(c) for c in crit]}")
                found=True; break
        if found: break
s3=sp.sqrt(3); s5=sp.sqrt(5)
al = sp.Symbol('al')
v22 = [r for r in sp.Poly(al**3+2*al**2-2,al).nroots(maxsteps=500)]
b5 = [r for r in sp.Poly(al**4+al**2-1,al).nroots(maxsteps=500)]
which = sys.argv[1:]
cases = {
 'B1': ((x+y+1)**6/(x*y**2*z)+z, 0, [24*s3,-24*s3]),
 'B2': ((x+y+1)**4/(x*y*z)+z, 0, [16,-16]),
 'B3': ((x+y+1)**3/(x*y*z)+z, 0, [6*s3,-6*s3]),
 'B4': ((x+1)**2*(y+1)**2/(x*y*z)+z, 0, [8,-8]),
 'B5': (x+y+z+1/x+1/y+1/z+1/(x*y*z), 0, [4*r**3+6*r for r in b5]),
 'V6': ((x+1)**2*(y+z+1)**3/(x*y*z), 12, [108-12]),
 'V10': ((1+x+y+z+x*y+x*z+y*z)**2/(x*y*z), 6, [22+10*s5-6, 22-10*s5-6]),
 'V14': ((x+y+z+1)**2/x + (x+y+z+1)*(y+z+1)*(z+1)**2/(x*y*z), 4, [27-4, -1-4]),
 'V18': ((x+y+z)*(x+y+z+x*y+x*z+y*z+x*y*z)/(x*y*z), 3, [9+6*s3-3, 9-6*s3-3]),
 'V22': (x*y/z + x + x/z + y + y/z + z + 4 + 1/z + z/y + 1/y + z/x + 1/x + z/(x*y) + 1/(x*y), 4, [3*r**2+8*r+8-4 for r in v22]+[0-4]),
}
for k in (which or cases):
    f,sh,cr = cases[k]; analyse(k, f, sh, [sp.N(c) for c in cr])
