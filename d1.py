from common import *
al = sp.Symbol('alpha')
print("=== B1 ===")
H = X*Y - X*T - T**2
F = X**4 + Z**2*H - lam*Y*Z*H
h = x*y - x - 1
fl = chart(F, T)
print("U_T form ok:", sp.expand(fl - (x**4 + (z**2 - lam*y*z)*h))==0)
# all singular points off the base locus V(F_0) cap V(YZH), on every chart, lam unknown
for var in (T,X,Y,Z):
    G = chart(F,var); vs = CHART_VARS[var]
    sec = chart(Y*Z*H, var)
    eqs = [G] + [sp.diff(G,v) for v in vs]
    gens = [sp.expand(e) for e in eqs] + [sp.expand(w_*sec - 1)]
    gb = sp.groebner(gens, w_, *vs, lam, order='lex')
    print(f" chart U_{var}: off-base singular ideal (elim w):", [g for g in gb.exprs if not g.has(w_)])
q = {x:-2, y:sp.Rational(1,3), z:4*al}
fq = fl.subs(lam, 24*al)
print("q_alpha check: f, grad mod alpha^2-3:", [sp.rem(sp.expand(e.subs(q)), al**2-3, al) for e in [fq]+[sp.diff(fq,v) for v in (x,y,z)]])
print("det Hess (x,y,z):", sp.rem(sp.expand(sp.hessian(fq,(x,y,z)).subs(q).det()), al**2-3, al))
print("projective q_alpha = [x:y:z:1]*3 =", [3*q[x], 3*q[y], 3*q[z], 3])
# standard B1 Laurent polynomial comparison
fs = (x+y+1)**6/(x*y**2*z) + z
eqs = [sp.numer(e) for e in logjac(fs)]
gb = torus_groebner(eqs); print("standard B1 Laurent (x+y+1)^6/(xy^2z)+z: saturated crit ideal", gb)
zz = sp.sqrt(432)
print("   critical values:", [sp.simplify(fs.subs({x:sp.Rational(1,3),y:sp.Rational(2,3),z:s*zz})) for s in (1,-1)], " vs 24*alpha =", [24*sp.sqrt(3), -24*sp.sqrt(3)])

print("=== V6 ===")
f6 = (x+1)**2*(y+z+1)**3/(x*y*z)
eqs = [sp.numer(sp.together(e)) for e in logjac(f6)]
print("saturated crit ideal (by xyz*(x+1)*(y+z+1)):", torus_groebner(eqs, extra_nonzero=(x+1)*(y+z+1)))
s = {x:1,y:1,z:1}; print("lambda =", f6.subs(s))
g = (x+1)**2*(y+z+1)**3 - 108*x*y*z
Hm = sp.hessian(g,(x,y,z)).subs(s); print("Hess:", Hm.tolist(), "det:", Hm.det(), sp.factorint(Hm.det()))
print("logHess det f6:", loghess(f6).subs(s).det())
