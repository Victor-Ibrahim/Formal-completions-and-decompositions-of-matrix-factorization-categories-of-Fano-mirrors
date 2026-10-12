import sympy as sp, re
from jumplib import sing, S, lam, x, y, z, mu_at
def jump_candidates_sat(f):
    out = sing(f'''LIB "elim.lib"; ring R=0,(x,y,z,lam),dp; poly f={S(f)};
ideal J = diff(f,x),diff(f,y),diff(f,z); def ss = sat(J, ideal(x,y,z)); ideal Js; if (typeof(ss)=="list") {{ Js = ss[1]; }} else {{ Js = ss; }}
ideal E = eliminate(std(Js+ideal(x,y,z)),x*y*z); "E",E[1];''')
    roots=set()
    for p in [l[2:] for l in out.splitlines() if l.startswith('E ')]:
        e = sp.sympify(p.replace('^','**'), locals={'lam':lam})
        if e==0: roots.add('ALL'); continue
        if e.is_number: continue
        for fac,_ in sp.factor_list(sp.Poly(e,lam))[1]: roots.add(fac.as_expr())
    return roots
a,b,u,v,w = sp.symbols('a b u v w')
F = (a+b)**2*(u+v+w)**3 - lam*a*b*u*v*w
# points: (P1 point, P2 point); chart: a=1 or b=1 ; one of u,v,w = 1 ; local coords x (P1), y,z (P2)
pts = [((1,-1),(1,0,0)),((1,-1),(0,1,0)),((1,-1),(0,0,1)),
       ((1,0),(1,-1,0)),((1,0),(1,0,-1)),((1,0),(0,1,-1)),
       ((0,1),(1,-1,0)),((0,1),(1,0,-1)),((0,1),(0,1,-1))]
for p1,p2 in pts:
    i1 = 0 if p1[0]!=0 else 1; i2 = [k for k in range(3) if p2[k]!=0][0]
    s1 = {a:None,b:None}; P1v=[a,b]; P2v=[u,v,w]
    sub = {P1v[i1]:1, P1v[1-i1]: x + sp.Rational(p1[1-i1],p1[i1])}
    rest = [k for k in range(3) if k!=i2]
    sub[P2v[i2]] = 1
    sub[P2v[rest[0]]] = y + sp.Rational(p2[rest[0]],p2[i2]); sub[P2v[rest[1]]] = z + sp.Rational(p2[rest[1]],p2[i2])
    f = sp.expand(F.subs(sub, simultaneous=True))
    o = sing(f'LIB "sing.lib"; ring Sg=(0,lam),(x,y,z),ds; poly f={S(f)}; "MU",milnor(f);')
    mug = int(re.search(r'MU (\-?\d+)', o).group(1))
    res = [mu_at(f,q) if q!='ALL' else ('every','?') for q in jump_candidates_sat(f)]
    print(f"  {p1},{p2}: generic mu={mug}; candidates: {res}")
