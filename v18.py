from common import *
print("=== V18 ===")
f = (x+y+z)*(x+y+z+x*y+x*z+y*z+x*y*z)/(x*y*z)
F = (X+Y+Z)*((X+Y+Z)*T**2+(X*Y+X*Z+Y*Z)*T+X*Y*Z) - lam*X*Y*Z*T
print("homog check:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
A = x+y+z; Bp = x+y+z+x*y+x*z+y*z+x*y*z
eqs = [sp.numer(e) for e in logjac(f)]
G = torus_groebner(eqs, extra_nonzero=A*Bp)   # isolated points off central fibre
print("saturated (by xyz*A*B) log-critical ideal:", G)
sols = [{x:s0,y:s0,z:s0} for s0 in (sp.sqrt(3),-sp.sqrt(3))]
a = sp.Symbol('a')
for s in sols:
    val = sp.nsimplify(sp.simplify(f.subs(s)))
    print("  pt", s, " value", sp.radsimp(val), "=", sp.N(val), " logHess", sp.radsimp(sp.simplify(loghess(f).subs(s).det())))
print("claimed: lam = 18a+9 = 9+-6sqrt3:", [sp.N(9+6*sp.sqrt(3)), sp.N(9-6*sp.sqrt(3))])
# chart hessians mod 3a^2-1 at p_a=[1:1:1:a]
for var in (T,X,Y,Z):
    Fl = F.subs(lam, 18*a+9)
    G_ = chart(Fl,var); vs = CHART_VARS[var]
    P = {X:1,Y:1,Z:1,T:a}; names={x:X,y:Y,z:Z,t:T}
    pt = {v: P[names[v]]/P[var] for v in vs}
    grad = [sp.rem(sp.numer(sp.together(sp.diff(G_,v).subs(pt))), 3*a**2-1, a) for v in vs]
    d = sp.together(hess(G_,vs).subs(pt).det())
    dn = sp.rem(sp.expand(sp.numer(d)), 3*a**2-1, a); dd = sp.denom(d)
    print(f"  chart U_{var}: grad mod(3a^2-1) = {grad}, detHess = ({dn})/({dd}) mod 3a^2-1; nonzero at roots: {[sp.N((dn/dd).subs(a,r)) for r in (1/sp.sqrt(3),-1/sp.sqrt(3))]}")
# central fibre
A_ = X+Y+Z; B_ = (X+Y+Z)*T**2+(X*Y+X*Z+Y*Z)*T+X*Y*Z
print("B = (X+T)(Y+T)(Z+T) - T^3 ?", sp.expand(B_ - ((X+T)*(Y+T)*(Z+T)-T**3)))
print("C0 eq on A=0:", sp.factor(sp.expand(B_.subs(Z,-X-Y))))
