import os
"""For each smooth toric Fano threefold: paper's polynomial (correct label) vs
Hori-Vafa polynomial.  Critical points with multiplicity on the torus and number
of distinct critical values (= number of singular fibres on the torus)."""
import sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms, periods_mod
from toric import crit_data, CHI
from fans import FANS, hv

# actual Mori-Mukai family -> label under which the paper lists its polynomial
PAPER_LABEL = {k: k for k in FANS}
PAPER_LABEL.update({"MM_4_9": "MM_4_10", "MM_4_10": "MM_4_11",
                    "MM_4_11": "MM_4_12", "MM_4_12": "MM_4_13"})


def job(args):
    tag, expr = args
    t0 = time.time()
    mu, nd, vals = crit_data(expr)
    return tag, mu, nd, round(time.time() - t0, 1)


if __name__ == "__main__":
    models = all_models()
    P = 2147483629
    tasks = []
    for fam, rays in FANS.items():
        W = models[PAPER_LABEL[fam]]
        terms = laurent_terms(W)
        const = dict(terms).get((0, 0, 0), 0)
        H = hv(rays) + const
        ok = periods_mod(terms, 14, P) == periods_mod(laurent_terms(H), 14, P)
        print(f"{fam:8s} paper label {PAPER_LABEL[fam]:8s} periods match HV: {ok}", flush=True)
        tasks.append(((fam, "paper"), W))
        tasks.append(((fam, "HV"), H))
    res = {}
    with mp.Pool(4) as pool:
        its = [(t, pool.apply_async(job, (t,))) for t in tasks]
        for (tag, _), r in its:
            try:
                out = r.get(timeout=900)
                res[tag] = out[1:]
            except mp.TimeoutError:
                res[tag] = ("timeout", None, None)
            print(tag, res[tag], flush=True)
    print("\nfamily   chi | paper: mu  #values | HV: mu  #values")
    for fam in FANS:
        p, h = res[(fam, "paper")], res[(fam, "HV")]
        print(f"{fam:8s} {CHI[fam]:3d} | {str(p[0]):>8s} {str(p[1]):>8s} | {str(h[0]):>5s} {str(h[1]):>8s}")
