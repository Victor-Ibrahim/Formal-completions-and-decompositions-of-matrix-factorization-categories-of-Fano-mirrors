from common import *
print("=== d=2 source B2 ===")
FB = x*(y+z+1) + 1/(x*y*z); W = FB**2/x
FBn = sp.numer(sp.together(FB))   # x^2yz(y+z+1)+1
# smoothness of S_B2 on torus
print("S_B2 singular points on torus:", torus_groebner([FBn]+[sp.diff(FBn,v) for v in (x,y,z)]))
print("curve (t,-1,1/t) on S:", sp.simplify(FB.subs({x:t,y:-1,z:1/t})))
eqs = [sp.numer(e) for e in logjac(W)]
print("saturated by xyz*F:", torus_groebner(eqs, extra_nonzero=FBn))
for s in ({x:1,y:1,z:1},{x:-1,y:1,z:1}):
    print("  ", s, "W=", W.subs(s), " logHess det=", sp.simplify(loghess(W).subs(s).det()))

print("=== d=2 source V10 ===")
F10 = x*y*z**2+x*y*z+1; G10 = x*y*z**2+x*y*z+x*z+y*z+1
WV = sp.expand(F10**2*G10/(x**2*y**2*z**3)) - 6
print("number of monomials of W_V10:", len(sp.Add.make_args(WV)), "; of W_V10+6:", len(sp.Add.make_args(sp.expand(WV+6))))
print("W_V10 =", WV)
print("constant term of W_V10:", sum(m for m in sp.Add.make_args(WV) if not m.free_symbols))
print("K: G10-F10 =", sp.factor(G10-F10), "; F10|_{y=-x} =", sp.factor(F10.subs(y,-x)))
Jd = sp.Matrix([[sp.diff(F10,x),sp.diff(F10,y)],[sp.diff(G10,x),sp.diff(G10,y)]]).det()
print("det d(F,G)/d(x,y) on K:", sp.factor(sp.expand(Jd.subs(y,-x))))
print("S_V10 = V(F10) smooth on torus?", torus_groebner([F10]+[sp.diff(F10,v) for v in (x,y,z)]))
eqs = [sp.numer(sp.together(e)) for e in logjac(WV)]
Gs = torus_groebner(eqs, extra_nonzero=F10)
print("saturated (by xyz*F10) log-critical ideal:", Gs)
a = sp.Symbol('a')
P = {x:4*a-2, y:4*a-2, z:a}; m = 4*a**2-2*a-1
def red(e): 
    e = sp.together(e); n = sp.rem(sp.expand(sp.numer(e)), m, a); d = sp.expand(sp.denom(e))
    # invert denominator mod m
    inv = sp.invert(sp.Poly(d,a), sp.Poly(m,a)).as_expr()
    return sp.rem(sp.expand(n*inv), m, a)
print("crit eqs at P_a mod m:", [red(e.subs(P)) for e in eqs])
H = loghess(WV).applyfunc(lambda e: red(e.subs(P)))
print("W(P_a) mod m =", red(WV.subs(P)), " ; logHess det mod m =", sp.factor(sp.rem(sp.expand(H.det()), m, a)))

print("=== d=2 compactifiable B2 ===")
f = (x+y+1)**4/(x*y*z) + z
F = (X+Y+T)**4 + X*Y*Z**2 - lam*X*Y*Z*T
print("homog:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
eqs = [sp.numer(e) for e in logjac(f)]
print("saturated:", torus_groebner(eqs))
for zz in (8,-8):
    s = {x:sp.Rational(1,2),y:sp.Rational(1,2),z:zz}
    print("  ", s, "f=", f.subs(s), "logHess det=", loghess(f).subs(s).det())

print("=== d=2 compactifiable V10 ===")
q = 1+x+y+z+x*y+x*z+y*z
f = q**2/(x*y*z)
Q = T**2+T*(X+Y+Z)+X*Y+X*Z+Y*Z
F = Q**2 - lam*X*Y*Z*T
print("homog:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
eqs = [sp.numer(sp.together(e)) for e in logjac(f)]
print("saturated by xyz*q:", torus_groebner(eqs, extra_nonzero=q))
m = a**2+a-1; P = {x:1/a,y:1/a,z:1/a}
def red2(e):
    e = sp.together(e); n = sp.expand(sp.numer(e)); d = sp.expand(sp.denom(e))
    inv = sp.invert(sp.Poly(d,a), sp.Poly(m,a)).as_expr()
    return sp.rem(sp.expand(n*inv), m, a)
H = loghess(f).applyfunc(lambda e: red2(e.subs(P)))
print("f(r_a) mod a^2+a-1 =", red2(f.subs(P)), "; logHess det =", sp.factor(sp.rem(sp.expand(H.det()), m, a)))
M = sp.hessian(Q,(X,Y,Z,T))/2
print("det of Gram matrix of Q:", M.det())
