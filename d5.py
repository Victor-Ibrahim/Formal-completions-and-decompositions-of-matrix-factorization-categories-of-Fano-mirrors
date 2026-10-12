from common import *
al = sp.Symbol('alpha')
def redm(e, m, v=al):
    e = sp.together(e); n = sp.expand(sp.numer(e)); d = sp.expand(sp.denom(e))
    inv = sp.invert(sp.Poly(d,v), sp.Poly(m,v)).as_expr()
    return sp.rem(sp.expand(n*inv), m, v)
print("=== B5 ===")
F = X**2*Y*Z + X*Y**2*Z + X*Y*Z**2 + Y*Z*T**2 + X*Z*T**2 + X*Y*T**2 + T**4 - lam*X*Y*Z*T
f = sp.expand(chart(F,T).subs(lam,0)/(x*y*z))
print("Laurent f_B5 =", f)
eqs = [sp.numer(sp.together(e)) for e in logjac(f)]
gb = torus_groebner(eqs); print("saturated crit ideal:", gb)
# p_alpha = [1:1:1:alpha] ; U_X coords (y,z,t)=(1,1,alpha)
m = al**4+al**2-1
GX = chart(F.subs(lam, 4*al**3+6*al), X); P = {y:1,z:1,t:al}
print("U_X: F, grad at p_alpha mod m:", [sp.rem(sp.expand(e.subs(P)),m,al) for e in [GX]+[sp.diff(GX,v) for v in (y,z,t)]])
d = sp.rem(sp.expand(sp.hessian(GX,(y,z,t)).subs(P).det()), m, al)
print("det Hess (U_X) mod m =", sp.factor(d), " ; claimed 2(3a^2+4) ->", sp.expand(d-2*(3*al**2+4))==0)
print("critical values 4a^3+6a numerically:", [sp.N(4*r**3+6*r) for r in sp.Poly(m,al).nroots()])
print("value at diag via Laurent:", sp.factor(redm(f.subs({x:1/al,y:1/al,z:1/al}), m)))
print("=== V22 ===")
F = (X**2*Y**2 + X*Y**2*T + X**2*Y*T + X**2*Y*Z + X*Y**2*Z + X*Y*T**2 + 4*X*Y*Z*T + Y*Z*T**2 + X*Z*T**2
     + X*Y*Z**2 + Z*T**3 + Y*Z**2*T + X*Z**2*T + Z**2*T**2 - lam*X*Y*Z*T)
f = sp.expand(chart(F,T).subs(lam,0)/(x*y*z))
print("Laurent f_V22 =", f)
eqs = [sp.numer(sp.together(e)) for e in logjac(f)]
gb = torus_groebner(eqs); print("saturated crit ideal:", gb)
m = al**3+2*al**2-2
print("discriminant of cubic:", sp.discriminant(m, al), "(nonzero -> 3 distinct roots)")
G1 = chart(F.subs(lam, 3*al**2+8*al+8), X); P = {y:1,z:1,t:al}
print("U_X: F, grad at p_alpha mod m:", [sp.rem(sp.expand(e.subs(P)),m,al) for e in [G1]+[sp.diff(G1,v) for v in (y,z,t)]])
d = sp.rem(sp.expand(sp.hessian(G1,(y,z,t)).subs(P).det()), m, al)
print("det Hess on X=1 mod m:", sp.factor(d))
print("Res(cubic, 14a^2+39a+32) =", sp.resultant(m, 14*al**2+39*al+32, al))
print("critical values:", [sp.N(3*r**2+8*r+8) for r in sp.Poly(m,al).nroots()])
# base points
pts = {'P1^1':(1,0,0,0),'P1^2':(0,1,0,0),'P2^1':(1,0,0,-1),'P2^2':(0,1,0,-1),'P3':(0,0,1,0),'P4':(1,1,-1,0),'P5':(1,1,0,-1),'P5^1':(1,0,1,-1),'P5^2':(0,1,1,-1)}
F0 = F.subs(lam,0); G = X*Y*Z*T
for k,p in pts.items():
    s = dict(zip((X,Y,Z,T),p))
    g0 = [sp.diff(F0,v).subs(s) for v in (X,Y,Z,T)]; gG = [sp.diff(G,v).subs(s) for v in (X,Y,Z,T)]
    print(f"  {k} {p}: F0={F0.subs(s)}, G={G.subs(s)}, grad F0={g0}, grad G={gG}")
# boundary blowup on X=1: q=Y, u=Y+Z, t=T ; Z = u - q
q_,u_,r_ = sp.symbols('q u r')
F1 = sp.expand(chart(F,X).subs({y:q_, z:u_-q_}))   # chart X: coords (y,z,t)
tot = sp.expand(F1.subs(u_, t*r_))
print("total transform divisible by t:", sp.rem(sp.Poly(tot,t), sp.Poly(t,t)).as_expr()==0)
g = sp.expand(sp.cancel(tot/t))
g101 = (lam*q_**2 + q_**3 - q_**2*r_ - 2*q_**2 + q_*r_ + q_ + q_*r_*t*(-lam-2*q_+r_+2)
        + t**2*(q_*r_**2 - q_*r_ - q_ + r_**2 + r_) + r_*(r_+1)*t**3)
print("(101) matches:", sp.expand(g-g101)==0)
if sp.expand(g-g101)!=0: print("   difference:", sp.factor(sp.expand(g-g101)))
p = {q_:1, r_:0, t:0}
g0 = g.subs(lam,0)
print("g_0(1,0,0)=", g0.subs(p), "grad:", [sp.diff(g0,v).subs(p) for v in (q_,r_,t)])
Hg = sp.hessian(g0,(q_,r_,t)).subs(p); print("Hess g0:", Hg.tolist(), "det", Hg.det())
sec = sp.expand(sp.cancel(sp.expand((x*y*z*t).subs({x:1}).subs({y:q_, z:u_-q_}).subs(u_, t*r_))/t)) if False else sp.expand(sp.cancel(sp.expand((q_*(u_-q_)*t).subs(u_,t*r_))/t))
print("second generator strict transform:", sp.factor(sec), " value at p:", sec.subs(p))
print("generic lam: grad of g_lam at p:", [sp.factor(sp.diff(g,v).subs(p)) for v in (q_,r_,t)], " g_lam(p)=", g.subs(p))
