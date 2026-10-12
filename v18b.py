from common import *
r_,h_,tau,s = sp.symbols('r h tau s')
# p0 square completion: z = r - x - y, B = r + h
Bp = (x+y+z)+x*y+x*z+y*z+x*y*z
h = sp.expand(Bp.subs(z, r_-x-y) - r_)
print("h|_{r=0} =", sp.factor(h.subs(r_,0)), " quadratic part:", sp.Poly(h.subs(r_,0),x,y).as_expr() )
print("AB - [(r+h/2)^2-(h/2)^2] =", sp.expand(r_*(r_+h) - ((r_+h/2)**2-(h/2)**2)))
g = x**2+x*y+y**2+x*y*(x+y)
gt = sp.expand(g.subs({x:tau*s, y:s}))
print("blowup g:", sp.factor(gt), "-> strict:", sp.expand(sp.cancel(gt/s**2)))
print("roots at s=0:", sp.solve(sp.expand(sp.cancel(gt/s**2)).subs(s,0), tau))
