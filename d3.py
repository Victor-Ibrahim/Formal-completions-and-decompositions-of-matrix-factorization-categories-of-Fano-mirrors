from common import *
a = sp.Symbol('a')
def redm(e, m):
    e = sp.together(e); n = sp.expand(sp.numer(e)); d = sp.expand(sp.denom(e))
    inv = sp.invert(sp.Poly(d,a), sp.Poly(m,a)).as_expr()
    return sp.rem(sp.expand(n*inv), m, a)
print("=== source B3 ===")
W = x + x/(y*z) + y + z + 2/z + 2/y + y/(x*z) + 2/x + z/(x*y)
eqs = [sp.numer(sp.together(e)) for e in logjac(W)]
print("saturated:", torus_groebner(eqs))
P = {x:a,y:a,z:a}; m = a**2-3
H = loghess(W).applyfunc(lambda e: redm(e.subs(P),m))
print("W(a,a,a) =", redm(W.subs(P),m), " logHess det =", sp.rem(sp.expand(H.det()),m,a))

print("=== source V14 ===")
A = y*z**2+y*z+1
B = x**2*y*z**2 + x*y**2*z**4 + 2*x*y**2*z**3 + x*y**2*z**2 + 2*x*y*z + x + y*z**2
WV = sp.expand(A*B/(x*y**2*z**3)) - 4
print("W_V14 =", WV, "; #monomials", len(sp.Add.make_args(WV)))
print("B on A=0 :", sp.factor(sp.expand(B.subs(y, -1/(z*(z+1))))))
print("Gamma_src on A=0, x=1:", sp.simplify(A.subs({x:1,y:-1/(t*(t+1)),z:t})))
eqs = [sp.numer(sp.together(e)) for e in logjac(WV)]
print("saturated by xyz*A:", torus_groebner(eqs, extra_nonzero=A))
for s in ({x:1,y:1,z:1},{x:-1,y:9,z:sp.Rational(-1,3)}):
    print("  ", s, "W=", WV.subs(s), " logHess det=", loghess(WV).subs(s).det(), " A=", A.subs(s))

print("=== compactifiable B3 ===")
f = (x+y+1)**3/(x*y*z) + z
F = T*(X+Y+T)**3 + X*Y*Z**2 - lam*X*Y*Z*T
print("homog:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
eqs = [sp.numer(e) for e in logjac(f)]
print("saturated:", torus_groebner(eqs))
P = {x:1,y:1,z:3*a}
H = loghess(f).applyfunc(lambda e: redm(e.subs(P),m))
print("f(q_a) =", redm(f.subs(P),m), "; logHess det =", sp.rem(sp.expand(H.det()),m,a))

print("=== compactifiable V14 ===")
Ah = X+Y+Z+T; Bh = Y+Z+T; Ch = Z+T
f = (x+y+z+1)**2/x + (x+y+z+1)*(y+z+1)*(z+1)**2/(x*y*z)
F = Ah*(Y*Z*Ah + Bh*Ch**2) - lam*X*Y*Z*T
print("homog:", sp.simplify(F.subs({X:x,Y:y,Z:z,T:1}) - x*y*z*(f-lam)))
eqs = [sp.numer(sp.together(e)) for e in logjac(f)]
An = x+y+z+1; Cn = z+1
print("saturated by xyz*A*C:", torus_groebner(eqs, extra_nonzero=An*Cn))
s = {x:6,y:sp.Rational(3,2),z:sp.Rational(1,2)}
print("r:", s, "f=", f.subs(s), "logHess=", loghess(f).subs(s).det())
print("Gamma_cmp (t,-t,-1) critical? ", [sp.simplify(e.subs({x:t,y:-t,z:-1})) for e in logjac(f)], " f=", sp.simplify(f.subs({x:t,y:-t,z:-1})))
U,V = sp.symbols('U V')
f72 = U*(-U**2*V+U**2+(t-1)*U*V-t*U+t*V**2)/(t*(V-1)*(-U+V+t))
print("(72) check:", sp.simplify(f.subs({x:t, y:U-t-V, z:V-1}) - f72))
print("U(U-V^2) = (U-V^2/2)^2 - V^4/4:", sp.expand(U*(U-V**2) - ((U-V**2/2)**2 - V**4/4)))
# boundary node: chart U_Y, x=X/Y, z=Z/Y, t=T/Y, b=1+z+t
b,r_ = sp.symbols('b r')
FY = chart(F, Y)  # coords (x,z,t)
FYb = sp.expand(FY.subs(z, b-t-1))
print("base curve x=b=0 in base locus? F|_{x=b=0}:", sp.simplify(FYb.subs({x:0,b:0})))
tot = sp.expand(FYb.subs(x, b*r_))
st = sp.expand(sp.cancel(tot/b))
c73 = b*(r_+1)**2*(b-t-1) + b*(r_+1)*(b-1)**2 - lam*r_*(b-t-1)*t
print("(73) strict transform matches:", sp.expand(st - c73)==0, "; remainder is nonzero modulo b (i.e. not divisible by b again):", sp.rem(sp.Poly(st,b), sp.Poly(b,b)).as_expr()!=0)
vs = (b,r_,t); o = {b:0,r_:0,t:0}
print("grad at 0:", [sp.diff(c73,v).subs(o) for v in vs])
Hl = sp.hessian(c73, vs).subs(o)
print("H_lam =", Hl.tolist(), " det =", sp.factor(Hl.det()))
# lam=-1 blow-up of origin
e,R,TT = sp.symbols('e R TT')
g = sp.expand(c73.subs(lam,-1).subs({b:e, r_:e*R, t:e*TT}))
G = sp.expand(sp.cancel(g/e**2))
G74 = -R**2*TT*e**2 + R**2*e**2 - R**2*e - R*TT**2*e - R*TT*e - R*TT + R*e**2 - R - TT + e - 1
print("(74) strict transform matches:", sp.expand(G-G74)==0)
p = {e:0,R:-1,TT:-1}
print("G(p)=",G.subs(p)," grad=",[sp.diff(G,v).subs(p) for v in (e,R,TT)])
HG = sp.hessian(G,(e,R,TT)).subs(p); print("Hess G =", HG.tolist(), " det =", HG.det())
# is it off the base locus? second generator transform
P1 = r_*(b-t-1)*t
p1 = sp.expand(sp.cancel(sp.expand(P1.subs({b:e, r_:e*R, t:e*TT}))/e**2))
print("second generator strict transform at p:", p1.subs(p), "(nonzero => not base point)")
# the singular point for lam=-1 before blowup: milnor
