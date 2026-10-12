import sys, re, sympy as sp
from jumplib import *
def jump_candidates_sat(f):
    out = sing(f'''LIB "elim.lib"; ring R=0,(x,y,z,lam),dp; poly f={S(f)};
ideal J = diff(f,x),diff(f,y),diff(f,z); def ss = sat(J, ideal(x,y,z)); ideal Js; if (typeof(ss)=="list") {{ Js = ss[1]; }} else {{ Js = ss; }}
ideal E = eliminate(std(Js+ideal(x,y,z)),x*y*z); "E",E[1];''')
    polys = [l[2:] for l in out.splitlines() if l.startswith('E ')]
    roots=set()
    for p in polys:
        e = sp.sympify(p.replace('^','**'), locals={'lam':lam})
        if e==0: roots.add('ALL'); continue
        if e.is_number: continue
        for fac,_ in sp.factor_list(sp.Poly(e,lam))[1]: roots.add(fac.as_expr())
    return roots
for name in sys.argv[1:]:
    F0,Gn,pts = pencils[name]
    print("=====", name, flush=True)
    for pt,cv in pts:
        f = at(F0-lam*Gn, pt, cv)
        o = sing(f'LIB "sing.lib"; ring S=(0,lam),(x,y,z),ds; poly f={S(f)}; "MU",milnor(f);')
        mug = int(re.search(r'MU (\-?\d+)', o).group(1))
        res=[]
        for q in jump_candidates_sat(f):
            if q=='ALL': res.append(('every lam','?')); continue
            res.append(mu_at(f,q))
        print(f"  {pt}: generic mu={mug}; mu at candidate lam: {res}", flush=True)
