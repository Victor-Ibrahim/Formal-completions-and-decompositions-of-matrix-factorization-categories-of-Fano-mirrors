import sympy as sp
x,y,z,t,lam = sp.symbols('x y z t lam')
X,Y,Z,T = sp.symbols('X Y Z T')

def logjac(f, vs=(x,y,z)):
    return [sp.together(v*sp.diff(f,v)) for v in vs]

def loghess(f, vs=(x,y,z)):
    th = [lambda g,v=v: v*sp.diff(g,v) for v in vs]
    return sp.Matrix(3,3,lambda i,j: th[i](th[j](f)))

def hess(F, vs):
    return sp.hessian(F, vs)

def chart(F, var):
    """dehomogenize F(X,Y,Z,T) on chart var=1; returns poly in remaining affine coords
    naming: U_T -> (x,y,z); U_X -> (y,z,t); U_Y -> (x,z,t); U_Z -> (x,y,t)"""
    sub = {X:x, Y:y, Z:z, T:t}
    sub[var] = 1
    return sp.expand(F.subs(sub, simultaneous=True))

CHART_VARS = {T:(x,y,z), X:(y,z,t), Y:(x,z,t), Z:(x,y,t)}

w_ = sp.Symbol('w_')
def torus_solve(eqs, vs=(x,y,z), extra_nonzero=1):
    """solve polynomial eqs on torus (vs nonzero, and extra_nonzero != 0) via Rabinowitsch"""
    eqs = [sp.expand(e) for e in eqs] + [sp.expand(w_*sp.prod(vs)*extra_nonzero - 1)]
    sols = sp.solve(eqs, list(vs)+[w_], dict=True)
    out = []
    for s in sols:
        s = {k:v for k,v in s.items() if k!=w_}
        out.append(s)
    return out

def torus_groebner(eqs, vs=(x,y,z), extra_nonzero=1):
    eqs = [sp.expand(e) for e in eqs] + [sp.expand(w_*sp.prod(vs)*extra_nonzero - 1)]
    G = sp.groebner(eqs, w_, *vs, order='lex')
    return [g for g in G.exprs if not g.has(w_)]
