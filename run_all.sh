#!/bin/bash
# Reruns all checks. Requires: python3 + sympy, Singular (apt install singular).
cd "$(dirname "$0")"
run(){ echo; echo "################ $1"; shift; "$@" 2>&1 | grep -v "redefining"; }
run "d=4  B4 (torus, charts, Hessians)"              python3 b4.py
run "d=4  B4 (Jacobian radicals, local types)"      Singular -q b4.sing
run "d=4  V18 (torus, charts)"                      python3 v18.py
run "d=4  V18 (central fiber)"                     Singular -q v18.sing
run "d=4  V18 (node p0, blow-up d+-)"              python3 v18b.py
run "d=2  B2/V10 (source and compactifiable models)" python3 d2.py
run "d=2/3 central supports B2, B3, V10"            Singular -q b2.sing
run "d=2/3 transverse types"                        Singular -q tr.sing
run "d=3  B3/V14 (source, compactifiable, boundary node)" python3 d3.py
run "d=3  V14 Milnor number at the point in (73)"   Singular -q v14.sing
run "d=1  B1/V6"                                     python3 d1.py
run "d=1  B1 base-point types"                      Singular -q b1types.sing
run "d=5  B5/V22"                                    python3 d5.py
run "d=5  V22 types at P4, P5"                        Singular -q v22b.sing
run "d=5  V22/V18 types at jumps"                    Singular -q v22c.sing
run "B1/B5/V14 types (A7, A4, jump at P6)"           Singular -q types.sing
run "Hodge invariants: e(X), H^3, index, h^{1,2}, k" python3 hodge.py
run "Periods (Fano side vs Laurent)"                 python3 periods.py
run "Fano-side periods (V18 via Gr(5,7))"            python3 periods2.py
run "Picard-Fuchs singularities"                     python3 recur.py
for p in B1 B2 B3 B4 B5 V10 V14 V18 V22; do run "Milnor-number jumps at base points: $p" python3 jumps3.py $p; done
run "Milnor-number jumps at base points: V6"          python3 v6jumps.py
