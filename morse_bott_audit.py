#!/usr/bin/env python3
"""Exact symbolic audit of every Morse--Bott label in ``main.tex``.

The checks are performed in the algebraic torus.  For each retained
Morse--Bott component the script verifies that

* the proposed component lies in the logarithmic critical locus;
* the component is smooth of the asserted codimension; and
* the logarithmic Hessian has exactly that codimension as its rank.

The exceptional ``V_8`` surface is treated separately: its singular locus is
computed exactly, so the script also certifies why the whole surface is not a
Morse--Bott component.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import sympy as sp

from mori_mukai_validation import CASES, log_hessian, log_system


x, y, z = sp.symbols("x y z")
TORUS_VARS = (x, y, z)


@dataclass(frozen=True)
class MorseBottDatum:
    label: str
    potential: sp.Expr
    generators: tuple[sp.Expr, ...]
    codimension: int
    exclusions: tuple[sp.Expr, ...] = ()


def numerators(expressions: Iterable[sp.Expr]) -> list[sp.Expr]:
    return [sp.factor(sp.fraction(sp.together(expr))[0]) for expr in expressions]


def minors(matrix: sp.Matrix, size: int) -> list[sp.Expr]:
    return [
        sp.factor(matrix.extract(rows, columns).det())
        for rows in itertools.combinations(range(matrix.rows), size)
        for columns in itertools.combinations(range(matrix.cols), size)
    ]


def empty_on_open_torus(
    equations: Sequence[sp.Expr], exclusions: Sequence[sp.Expr] = ()
) -> bool:
    auxiliary = sp.symbols("T_audit")
    inverted = x * y * z * sp.prod(exclusions)
    basis = sp.groebner(
        numerators(equations) + [1 - auxiliary * inverted],
        auxiliary,
        x,
        y,
        z,
        order="lex",
    )
    return any(polynomial.as_expr() == 1 for polynomial in basis.polys)


def verify_morse_bott(datum: MorseBottDatum) -> None:
    ideal_generators = numerators(datum.generators)
    component_basis = sp.groebner(ideal_generators, x, y, z, order="grevlex")
    critical_equations = numerators(log_system(datum.potential, TORUS_VARS))
    if not all(
        sp.factor(component_basis.reduce(equation)[1]) == 0
        for equation in critical_equations
    ):
        raise AssertionError(f"{datum.label}: component is not contained in the critical locus")

    jacobian = sp.Matrix(ideal_generators).jacobian(TORUS_VARS)
    singular_equations = minors(jacobian, datum.codimension)
    if not empty_on_open_torus(
        [*ideal_generators, *singular_equations], datum.exclusions
    ):
        raise AssertionError(f"{datum.label}: component is not smooth everywhere")

    hessian = log_hessian(datum.potential, TORUS_VARS)
    if datum.codimension == 1:
        rank_drop_equations = list(hessian)
    else:
        rank_drop_equations = minors(hessian, datum.codimension)
    if not empty_on_open_torus(
        [*ideal_generators, *rank_drop_equations], datum.exclusions
    ):
        raise AssertionError(
            f"{datum.label}: transverse logarithmic Hessian drops rank"
        )

    print(f"PASS {datum.label}: smooth, critical, transverse rank {datum.codimension}")


def build_retained_data() -> list[MorseBottDatum]:
    # Picard-rank-one B_2: W=F^2/x and S=V(F).
    f_b2 = x * (y + z + 1) + 1 / (x * y * z)
    w_b2 = f_b2**2 / x

    # Picard-rank-one B_4 source model: one smooth Morse--Bott curve.
    w_b4 = (
        x
        + y * z**2
        + 2 * y * z
        + y
        + 2 / (y * z)
        + 1 / (x * y**2 * z**2)
    )

    # Equations for the Mori--Mukai components.  They are extracted from the
    # saturated critical ideals or the parametric certificates in the
    # companion validation script.
    f208 = x * (y + z + 1) + 1 + 1 / (y * z)
    f228 = y**2 * z - y * z**2 - y * z - 1
    g228 = (
        2 * x
        - y**2 * z**4
        + y**2 * z**3
        - y**2 * z**2
        + y**2 * z
        + y * z**5
        - y * z
        + 2 * y
        + z**3
        - z**2
        - z
        - 3
    )
    g301 = (
        3 * x**2
        + 3 * x * y * z
        - 6 * x * y
        - 6 * x
        - 2 * y**3 * z
        + y**3
        - y**2 * z
        + 2 * y**2
        - y * z
        + 5 * y
        + z
        + 1
    )
    f701 = x**2 * z**2 + x**2 * z + 2 * x * z + 1

    return [
        MorseBottDatum("B_2/S", w_b2, (f_b2,), 1),
        MorseBottDatum("B_4/Gamma", w_b4, (x * y - 1, z + 1), 2),
        MorseBottDatum("MM_2-8/Sigma", CASES["MM_2_8"].potential, (f208,), 1),
        MorseBottDatum(
            "MM_2-15/Gamma", CASES["MM_2_15"].potential, (z + 1, x + y - 1), 2
        ),
        MorseBottDatum(
            "MM_2-28/Gamma", CASES["MM_2_28"].potential, (f228, g228), 2
        ),
        MorseBottDatum(
            "MM_3-1/Gamma_3", CASES["MM_3_1"].potential, (x - 1, y + 1), 2
        ),
        MorseBottDatum(
            "MM_3-1/Gamma_zeta",
            CASES["MM_3_1"].potential,
            (y**2 + y + 1, g301),
            2,
        ),
        MorseBottDatum(
            "MM_4-1/Gamma",
            CASES["MM_4_1"].potential,
            (x * z - 1, y * z + y + 1),
            2,
        ),
        MorseBottDatum(
            "MM_4-3/Gamma",
            CASES["MM_4_3"].potential,
            (z + 1, x * y - x + y),
            2,
        ),
        MorseBottDatum(
            "MM_7-1/Gamma_1", CASES["MM_7_1"].potential, (y - 1, f701), 2
        ),
        MorseBottDatum(
            "MM_7-1/Gamma_-1", CASES["MM_7_1"].potential, (y + 1, f701), 2
        ),
    ]


def verify_explicit_graph_components() -> None:
    """Check the components supplied with global graph/normal coordinates."""
    t, u, v = sp.symbols("t u v")

    # MM_2-5: the existing expansion certificate gives the quadratic form in
    # the global normal generators U=xy-(x^3+x^2-1) and V=xz+y^2.
    check_205 = CASES["MM_2_5"].expansion_checks[0]
    q_205 = check_205.expected + 6
    det_205 = sp.factor(sp.hessian(q_205, check_205.normal_variables).det())
    expected_205 = -t**6 / (t**3 + t**2 - 1) ** 6
    if sp.factor(det_205 - expected_205) != 0:
        raise AssertionError("MM_2-5/Gamma: transverse determinant mismatch")
    print(f"PASS MM_2-5/Gamma: global normal determinant {expected_205}")

    # MM_2-6: both displayed quadratic expansions have unit determinant on
    # their parameter spaces.
    for index, expected in enumerate((4 * t, -t**2 / 4), start=1):
        expansion = CASES["MM_2_6"].expansion_checks[index - 1]
        q = expansion.expected + 4
        determinant = sp.factor(sp.hessian(q, expansion.normal_variables).det())
        if sp.factor(determinant - expected) != 0:
            raise AssertionError(f"MM_2-6/Gamma_{index}: determinant mismatch")
        print(f"PASS MM_2-6/Gamma_{index}: global normal determinant {expected}")

    def ordinary_normal_determinant(
        case_name: str,
        normal_indices: tuple[int, int],
        substitution: dict[sp.Symbol, sp.Expr],
    ) -> sp.Expr:
        hessian = sp.hessian(CASES[case_name].potential, TORUS_VARS)
        normal_block = hessian.extract(normal_indices, normal_indices).subs(substitution)
        return sp.factor(sp.together(normal_block.det()))

    d207 = t**2 + t + 1
    graph_checks = (
        (
            "MM_2-7/Gamma",
            "MM_2_7",
            (0, 2),
            {x: -d207 / (t * (t + 1) ** 2), y: t, z: -d207**2 / t**2},
            -t**6 * (t + 1) ** 4 / d207**6,
        ),
        ("MM_2-12/Gamma", "MM_2_12", (0, 2), {x: 1, y: t, z: -1}, 4 * t),
        ("MM_2-13/Gamma", "MM_2_13", (0, 1), {x: 1, y: -1, z: t}, -5),
        ("MM_2-19/Gamma", "MM_2_19", (0, 1), {x: -1, y: 1, z: t}, 3),
    )
    for label, case_name, indices, substitution, expected in graph_checks:
        determinant = ordinary_normal_determinant(case_name, indices, substitution)
        if sp.factor(determinant - expected) != 0:
            raise AssertionError(f"{label}: determinant mismatch")
        print(f"PASS {label}: global normal determinant {expected}")

    # MM_3-2 has an exact split form W+6=rs and an invertible coordinate
    # Jacobian along the curve.
    w302 = CASES["MM_3_2"].potential
    u302 = 1 + (x + y) / (x * y * z)
    r302 = z * u302
    s302 = x * y + 1 + x * y * z * u302**2
    if sp.factor(sp.together(w302 + 6 - r302 * s302)) != 0:
        raise AssertionError("MM_3-2/Gamma: exact split identity failed")
    jacobian = sp.factor(
        sp.Matrix([x, r302, s302])
        .jacobian(TORUS_VARS)
        .det()
        .subs({x: t, y: -1 / t, z: t - 1 / t})
    )
    if jacobian != -t:
        raise AssertionError("MM_3-2/Gamma: coordinate Jacobian mismatch")
    print("PASS MM_3-2/Gamma: exact W+6=rs and coordinate Jacobian -t")


def verify_v8_exception() -> None:
    f_v8 = (
        x**2 * y**2 * z**2
        + x**2 * y * z**2 * (z + 1) ** 2
        + 2 * x * y * z
        + 1
    )
    w_v8 = f_v8**2 / (x**3 * y**2 * z**4) - 8

    singular_equations = [
        f_v8,
        sp.diff(f_v8, x),
        sp.diff(f_v8, y),
        sp.diff(f_v8, z),
    ]
    if empty_on_open_torus(singular_equations):
        raise AssertionError("V_8/S: surface was incorrectly found smooth")

    # On the torus the singular locus is exactly C=V(xy-1,z+1).
    singular_numerators = numerators(singular_equations)
    singular_basis = sp.groebner(singular_numerators, x, y, z, order="grevlex")
    curve_basis = sp.groebner([x * y - 1, z + 1], x, y, z, order="grevlex")
    if not all(curve_basis.reduce(g)[1] == 0 for g in singular_numerators):
        raise AssertionError("V_8/S: the proposed singular curve is too small")
    if not all(singular_basis.reduce(g)[1] == 0 for g in (x * y - 1, z + 1)):
        raise AssertionError("V_8/S: the proposed singular curve is too large")

    t, a, b = sp.symbols("t a b")
    transverse_f = sp.expand(
        f_v8.subs({x: t, y: (1 + a) / t, z: b - 1})
    )
    transverse_hessian = sp.hessian(transverse_f, (a, b)).subs({a: 0, b: 0})
    if sp.factor(transverse_hessian.det()) != 4 * t:
        raise AssertionError("V_8/C: unexpected transverse Hessian determinant")

    point = {x: 1, y: 1, z: 1}
    if sp.simplify(w_v8.subs(point)) != 56:
        raise AssertionError("V_8: isolated critical value mismatch")
    if sp.factor(sp.hessian(w_v8, TORUS_VARS).det().subs(point)) != 32768:
        raise AssertionError("V_8: isolated point is not certified Morse")

    print("EXPECTED FAIL V_8/S: singular locus C=V(xy-1,z+1); not Morse--Bott")
    print("PASS V_8/C: F has transverse quadratic Hessian determinant 4*t")
    print("PASS V_8/(1,1,1): isolated Morse point, value 56, Hessian determinant 32768")


def main() -> None:
    for datum in build_retained_data():
        verify_morse_bott(datum)
    verify_explicit_graph_components()
    verify_v8_exception()


if __name__ == "__main__":
    main()