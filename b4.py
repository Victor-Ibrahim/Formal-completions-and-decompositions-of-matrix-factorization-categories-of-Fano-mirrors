from common import *
print("=== B4 ===")
f = (x+1)**2*(y+1)**2/(x*y*z) + z
# log critical points
eqs = [sp.numer(e) for e in logjac(f)]
sols = torus_solve(eqs)
print("saturated ideal:", torus_groebner(eqs))
print("log-critical points on torus:", sols)
for s in sols:
    print("  value", sp.simplify(f.subs(s)), " logHess det", sp.simplify(loghess(f).subs(s).det()))
# homogenization check (2)
F = (X+T)**2*(Y+T)**2 + X*Y*Z**2 - lam*X*Y*Z*T
print("homog check:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
# Hessians in each chart at p+- = [1:1:+-4:1]
for sgn in (1,-1):
    P = {X:1,Y:1,Z:4*sgn,T:1}; L = 8*sgn
    Fl = F.subs(lam,L)
    for var in (T,X,Y,Z):
        G = chart(Fl,var); vs = CHART_VARS[var]
        pt = {}
        names = {x:X,y:Y,z:Z,t:T}
        for v in vs: pt[v] = sp.Rational(P[names[v]], P[var])
        grad = [sp.diff(G,v).subs(pt) for v in vs]
        print(f"  p{'+' if sgn>0 else '-'} chart U_{var}: coords {tuple(pt.values())}, G={G.subs(pt)}, grad={grad}, detHess={hess(G,vs).subs(pt).det()}")
