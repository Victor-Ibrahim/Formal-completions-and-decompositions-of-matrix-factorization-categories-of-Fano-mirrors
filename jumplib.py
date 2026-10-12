import subprocess, sympy as sp, re, sys
lam = sp.Symbol('lam'); x,y,z = sp.symbols('x y z')
def sing(script):
    r = subprocess.run(['Singular','-q'], input=script+'\nquit;\n', capture_output=True, text=True)
    if '?' in r.stdout or r.stderr: raise RuntimeError(r.stdout[:2000]+r.stderr[:500])
    return r.stdout
def S(e): return str(sp.expand(e)).replace('**','^')

def analyze(F, name, skip_zero=True):
    """F: sympy poly in x,y,z,lam with p(lam) at origin."""
    out = sing(f'''LIB "sing.lib"; ring S=(0,lam),(x,y,z),ds; poly f={S(F)};
int m = milnor(f); "MU",m; ideal sb = std(jacob(f)); int i; poly g;
for(i=1;i<=size(sb);i++){{ g=sb[i]; while(g!=0){{ "C",leadcoef(g); g=g-lead(g);}} }}''')
    mu = int(re.search(r'MU (\-?\d+)', out).group(1))
    coeffs = [l[2:] for l in out.splitlines() if l.startswith('C ')]
    pr = set()
    for c in coeffs:
        e = sp.sympify(c.replace('^','**'), locals={'lam':lam})
        n,d = sp.fraction(sp.together(e))
        for part in (n,d):
            for fac,_ in sp.factor_list(sp.Poly(part,lam))[1] if sp.Poly(part,lam).degree()>0 else []:
                pr.add(fac.as_expr())
    res = []
    for q in sorted(pr, key=lambda e: sp.Poly(e,lam).degree()):
        P = sp.Poly(q,lam)
        if P.degree()==1:
            r0 = sp.solve(q,lam)[0]
            if skip_zero and r0==0: continue
            Fs = F.subs(lam, r0)
            o = sing(f'LIB "sing.lib"; ring S=0,(x,y,z),ds; poly f={S(Fs)}; "MU",milnor(f);')
        else:
            Fs = F.subs(lam, sp.Symbol('a'))
            o = sing(f'LIB "sing.lib"; ring S=(0,a),(x,y,z),ds; minpoly={S(q.subs(lam,sp.Symbol("a")))}; poly f={S(Fs)}; "MU",milnor(f);')
            r0 = f'root of {q}'
        m = int(re.search(r'MU (\-?\d+)', o).group(1))
        if m != mu: res.append((r0, m))
    print(f"  {name}: generic mu = {mu}; jumps (lam != 0): {res if res else 'none'}")
    return mu, res

X,Y,Z,T = sp.symbols('X Y Z T')
def at(Fh, point, chartvar):
    """dehomogenize Fh on chartvar=1 and shift so that point (tuple in X,Y,Z,T, may depend on lam) is origin"""
    P = dict(zip((X,Y,Z,T), [sp.sympify(p) for p in point])); c = P[chartvar]
    others = [v for v in (X,Y,Z,T) if v!=chartvar]
    sub = {chartvar:1}
    for v,new in zip(others,(x,y,z)): sub[v] = new + sp.simplify(P[v]/c)
    return sp.expand(Fh.subs(sub, simultaneous=True))

G = X*Y*Z*T
pencils = {
 'B4': ((X+T)**2*(Y+T)**2 + X*Y*Z**2, G,
        [((1,0,0,0),X),((0,1,0,0),Y),((0,0,1,0),Z),((-1,0,0,1),T),((0,-1,0,1),T),((-1,0,lam,1),T),((0,-1,lam,1),T)]),
 'V18': ((X+Y+Z)*((X+Y+Z)*T**2+(X*Y+X*Z+Y*Z)*T+X*Y*Z), G,
        [((1,-1,0,0),X),((1,0,0,0),X),((1,0,-1,0),X),((0,1,-1,0),Y),((0,1,0,0),Y),((0,0,1,0),Z),((0,0,0,1),T)]),
 'B2': ((X+Y+T)**4 + X*Y*Z**2, G,
        [((1,-1,0,0),X),((1,0,0,-1),X),((0,1,0,-1),Y),((0,0,1,0),Z),((-1,0,lam,1),T),((0,-1,lam,1),T)]),
 'V10': ((T**2+T*(X+Y+Z)+X*Y+X*Z+Y*Z)**2, G,
        [((1,0,0,0),X),((1,0,0,-1),X),((0,1,0,0),Y),((0,1,0,-1),Y),((0,0,1,0),Z),((0,0,1,-1),Z)]),
 'B3': (T*(X+Y+T)**3 + X*Y*Z**2, G,
        [((1,-1,0,0),X),((1,0,0,-1),X),((0,1,0,-1),Y),((0,0,1,0),Z),((-1,0,lam,1),T),((0,-1,lam,1),T)]),
 'V14': ((X+Y+Z+T)*(Y*Z*(X+Y+Z+T)+(Y+Z+T)*(Z+T)**2), G,
        [((1,-1,0,0),X),((1,0,0,0),X),((0,1,-1,0),Y),((0,1,0,-1),Y),((0,0,1,-1),Z),((lam,0,-1,1),T)]),
 'B5': (X**2*Y*Z+X*Y**2*Z+X*Y*Z**2+Y*Z*T**2+X*Z*T**2+X*Y*T**2+T**4, G,
        [((1,0,0,0),X),((0,1,0,0),Y),((0,0,1,0),Z),((1,-1,0,0),X),((1,0,-1,0),X),((0,1,-1,0),Y)]),
 'V22': (X**2*Y**2+X*Y**2*T+X**2*Y*T+X**2*Y*Z+X*Y**2*Z+X*Y*T**2+4*X*Y*Z*T+Y*Z*T**2+X*Z*T**2+X*Y*Z**2+Z*T**3+Y*Z**2*T+X*Z**2*T+Z**2*T**2, G,
        [((1,0,0,0),X),((1,0,0,-1),X),((0,1,0,0),Y),((0,1,0,-1),Y),((0,0,1,0),Z)]),
 'B1': (X**4 + Z**2*(X*Y-X*T-T**2), Y*Z*(X*Y-X*T-T**2),
        [((0,0,0,1),T),((0,1,0,0),Y),((0,0,1,0),Z),((0,1,lam,0),Y)]),
}


def mu_at(f, q):
    P = sp.Poly(q, lam)
    if P.degree()==1:
        r0 = sp.solve(q,lam)[0]
        o = sing(f'LIB "sing.lib"; ring S=0,(x,y,z),ds; poly f={S(f.subs(lam,r0))}; "MU",milnor(f);')
        lab = str(r0)
    else:
        a = sp.Symbol('a')
        o = sing(f'LIB "sing.lib"; ring S=(0,a),(x,y,z),ds; minpoly={S(q.subs(lam,a))}; poly f={S(f.subs(lam,a))}; "MU",milnor(f);')
        lab = f"root of {q}"
    return lab, int(re.search(r'MU (\-?\d+)', o).group(1))

