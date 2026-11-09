import os
"""(-K)^3 of the toric degeneration attached to a Fano polytope P:
6 * vol(P*), P* the polar dual.  Invariant under mutation."""
import sys
from fractions import Fraction
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibers import all_models, laurent_terms
from toric import TORIC
from fans import FANS

MM_DEGREE = {"P3": 64, "MM_2_33": 54, "MM_2_34": 54, "MM_2_35": 56, "MM_2_36": 62,
             "MM_3_25": 44, "MM_3_26": 46, "MM_3_27": 48, "MM_3_28": 48, "MM_3_29": 50,
             "MM_3_30": 50, "MM_3_31": 52, "MM_4_9": 40, "MM_4_10": 42, "MM_4_11": 44,
             "MM_4_12": 46, "MM_5_2": 36, "MM_5_3": 36}


def degree(points):
    pts = np.array(sorted(set(points)), dtype=float)
    hull = ConvexHull(pts)
    # facets a.v <= b with b > 0 (origin interior); dual vertices a/b
    duals = []
    for eq in hull.equations:          # n.v + off <= 0
        nrm, off = eq[:3], eq[3]
        duals.append(nrm / (-off))
    dual = ConvexHull(np.array(duals))
    return 6 * dual.volume


if __name__ == "__main__":
    models = all_models()
    for lab in TORIC:
        terms = laurent_terms(models[lab])
        pts = [e for e, _ in terms if e != (0, 0, 0)]
        dW = degree(pts)
        dF = degree(FANS[lab])
        print(f"{lab:8s} MM-degree={MM_DEGREE[lab]:2d}  paper-polytope={dW:6.2f}  my-fan={dF:6.2f}")
