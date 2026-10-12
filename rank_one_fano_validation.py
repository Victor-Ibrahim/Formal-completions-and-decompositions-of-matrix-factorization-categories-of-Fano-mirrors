#!/usr/bin/env python3
"""Exact symbolic validation for the 17 Picard-rank-one Fano threefold models.

The checks mirror the computations recorded in ``Apendice.tex``.  They cover
logarithmic critical equations, isolated points and Hessians, positive-
dimensional critical branches, compact factorizations, and the formal
quadratic expansions used in the categorical decompositions.

Run with, for example,

    python computations/rank_one_fano_validation.py --list
    python computations/rank_one_fano_validation.py --case V_12
    python computations/rank_one_fano_validation.py --all
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence

import sympy as sp


x, y, z = sp.symbols("x y z")
t, a, b, s = sp.symbols("t a b s")
U, V = sp.symbols("U V")
VARS = (x, y, z)


class CheckError(AssertionError):
    pass


def theta(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    return sp.factor(variable * sp.diff(expr, variable))


def log_system(potential: sp.Expr) -> tuple[sp.Expr, ...]:
    return tuple(theta(potential, variable) for variable in VARS)


def log_hessian(potential: sp.Expr) -> sp.Matrix:
    return sp.Matrix(
        [[theta(theta(potential, column), row) for column in VARS] for row in VARS]
    )


def ordinary_hessian(potential: sp.Expr) -> sp.Matrix:
    return sp.hessian(potential, VARS)


def assert_equal(label: str, actual: sp.Expr, expected: sp.Expr = sp.Integer(0)) -> None:
    difference = sp.factor(sp.cancel(actual - expected))
    if difference != 0:
        raise CheckError(
            f"{label}: expected {expected}, got {actual}; difference {difference}"
        )


def cleared_numerators(expressions: Iterable[sp.Expr]) -> list[sp.Expr]:
    numerators = []
    for expression in expressions:
        numerator, _denominator = sp.fraction(sp.together(expression))
        numerators.append(sp.factor(numerator))
    return numerators


def groebner_basis(
    expressions: Sequence[sp.Expr], variables: Sequence[sp.Symbol] = VARS
) -> sp.GroebnerBasis:
    return sp.groebner([sp.factor(expression) for expression in expressions], *variables, order="lex")


def ideals_equal(
    left: Sequence[sp.Expr],
    right: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol] = VARS,
) -> bool:
    left_basis = groebner_basis(left, variables)
    right_basis = groebner_basis(right, variables)
    return all(right_basis.reduce(generator)[1] == 0 for generator in left) and all(
        left_basis.reduce(generator)[1] == 0 for generator in right
    )


def saturated_generators(
    expressions: Sequence[sp.Expr], saturation_element: sp.Expr
) -> list[sp.Expr]:
    auxiliary = sp.symbols("T_rank_one_sat")
    basis = sp.groebner(
        [sp.factor(expression) for expression in expressions]
        + [1 - auxiliary * saturation_element],
        auxiliary,
        *VARS,
        order="lex",
    )
    return [
        sp.factor(polynomial.as_expr())
        for polynomial in basis.polys
        if not polynomial.as_expr().has(auxiliary)
    ]


def assert_saturated_ideal(
    label: str,
    expressions: Sequence[sp.Expr],
    expected_generators: Sequence[sp.Expr],
    saturation_element: sp.Expr = x * y * z,
) -> None:
    computed = saturated_generators(
        cleared_numerators(expressions), saturation_element
    )
    if not ideals_equal(computed, expected_generators):
        raise CheckError(
            f"{label}: saturated ideal differs from expectation\n"
            f"computed={computed}\nexpected={list(expected_generators)}"
        )


def rational_mod_equal(
    actual: sp.Expr,
    expected: sp.Expr,
    modulus: sp.Expr,
    parameter: sp.Symbol,
) -> bool:
    numerator, denominator = sp.fraction(sp.together(actual - expected))
    modulus_poly = sp.Poly(modulus, parameter)
    numerator_remainder = sp.rem(sp.Poly(numerator, parameter), modulus_poly).as_expr()
    if sp.factor(numerator_remainder) != 0:
        return False
    denominator_remainder = sp.rem(
        sp.Poly(denominator, parameter), modulus_poly
    ).as_expr()
    return sp.gcd(
        sp.Poly(denominator_remainder, parameter), modulus_poly
    ).degree() == 0


def coprime_mod(expr: sp.Expr, modulus: sp.Expr, parameter: sp.Symbol) -> bool:
    numerator, denominator = sp.fraction(sp.together(expr))
    modulus_poly = sp.Poly(modulus, parameter)
    numerator_remainder = sp.rem(sp.Poly(numerator, parameter), modulus_poly).as_expr()
    denominator_remainder = sp.rem(
        sp.Poly(denominator, parameter), modulus_poly
    ).as_expr()
    return (
        sp.gcd(sp.Poly(numerator_remainder, parameter), modulus_poly).degree() == 0
        and sp.gcd(sp.Poly(denominator_remainder, parameter), modulus_poly).degree()
        == 0
    )


def check_exact_point(
    label: str,
    potential: sp.Expr,
    point: dict[sp.Symbol, sp.Expr],
    value: sp.Expr,
    *,
    expected_log_hessian_det: sp.Expr | None = None,
    expected_ordinary_hessian_det: sp.Expr | None = None,
    expected_log_hessian_matrix: sp.Matrix | None = None,
) -> None:
    for coordinate in point.values():
        if sp.simplify(coordinate) == 0:
            raise CheckError(f"{label}: point is not on the algebraic torus")
    for equation in log_system(potential):
        assert_equal(f"{label} logarithmic critical equation", equation.subs(point))
    assert_equal(f"{label} critical value", potential.subs(point), value)

    if expected_log_hessian_matrix is not None:
        actual_matrix = log_hessian(potential).subs(point).applyfunc(sp.factor)
        if actual_matrix != expected_log_hessian_matrix:
            raise CheckError(
                f"{label}: logarithmic Hessian matrix mismatch\n"
                f"{actual_matrix}\n!=\n{expected_log_hessian_matrix}"
            )

    log_determinant = sp.factor(log_hessian(potential).det().subs(point))
    if expected_log_hessian_det is not None:
        assert_equal(
            f"{label} logarithmic Hessian determinant",
            log_determinant,
            expected_log_hessian_det,
        )
    elif log_determinant == 0:
        raise CheckError(f"{label}: logarithmic Hessian determinant vanished")

    if expected_ordinary_hessian_det is not None:
        ordinary_determinant = sp.factor(ordinary_hessian(potential).det().subs(point))
        assert_equal(
            f"{label} ordinary Hessian determinant",
            ordinary_determinant,
            expected_ordinary_hessian_det,
        )


def check_parametric_points(
    label: str,
    potential: sp.Expr,
    point: dict[sp.Symbol, sp.Expr],
    parameter: sp.Symbol,
    modulus: sp.Expr,
    value: sp.Expr,
    *,
    expected_log_hessian_det: sp.Expr | None = None,
) -> None:
    for coordinate in point.values():
        if not coprime_mod(coordinate, modulus, parameter):
            raise CheckError(f"{label}: coordinate {coordinate} vanishes on a branch")
    for equation in log_system(potential):
        if not rational_mod_equal(equation.subs(point), 0, modulus, parameter):
            raise CheckError(
                f"{label}: logarithmic critical equation does not vanish modulo {modulus}"
            )
    if not rational_mod_equal(potential.subs(point), value, modulus, parameter):
        raise CheckError(f"{label}: critical value mismatch modulo {modulus}")

    determinant = sp.factor(log_hessian(potential).det().subs(point))
    if expected_log_hessian_det is not None and not rational_mod_equal(
        determinant, expected_log_hessian_det, modulus, parameter
    ):
        raise CheckError(f"{label}: logarithmic Hessian determinant mismatch")
    if not coprime_mod(determinant, modulus, parameter):
        raise CheckError(f"{label}: logarithmic Hessian degenerates on a branch")


def check_branch(
    label: str,
    potential: sp.Expr,
    substitution: dict[sp.Symbol, sp.Expr],
    value: sp.Expr,
    exclusions: Sequence[sp.Expr] = (),
) -> None:
    for exclusion in exclusions:
        if sp.factor(sp.together(exclusion.subs(substitution))) == 0:
            raise CheckError(f"{label}: exclusion {exclusion} vanishes identically")
    for equation in log_system(potential):
        assert_equal(f"{label} logarithmic critical equation", equation.subs(substitution))
    assert_equal(f"{label} critical value", potential.subs(substitution), value)


def divide_log_system(
    label: str, potential: sp.Expr, common_factor: sp.Expr
) -> tuple[sp.Expr, ...]:
    quotients = []
    for variable, equation in zip(VARS, log_system(potential)):
        quotient = sp.cancel(equation / common_factor)
        assert_equal(f"{label} theta_{variable} common factor", equation, common_factor * quotient)
        quotients.append(quotient)
    return tuple(quotients)


def total_degree_truncation(
    expression: sp.Expr, normal_variables: Sequence[sp.Symbol], order: int
) -> sp.Expr:
    epsilon = sp.symbols("epsilon_rank_one")
    scaled = sp.together(
        expression.subs({variable: epsilon * variable for variable in normal_variables})
    )
    truncated = sp.series(scaled, epsilon, 0, order).removeO()
    return sp.factor(sp.together(truncated.subs(epsilon, 1)))


def check_expansion(
    label: str,
    expression: sp.Expr,
    expected: sp.Expr,
    normal_variables: Sequence[sp.Symbol] = (U, V),
    order: int = 3,
) -> None:
    low_terms = total_degree_truncation(expression - expected, normal_variables, order)
    assert_equal(label, low_terms)


def validate_P3() -> None:
    potential = x + y + z + 1 / (x * y * z)
    assert_saturated_ideal(
        "P3 critical ideal",
        log_system(potential),
        (x - y, y - z, z**4 - 1),
    )
    check_parametric_points(
        "P3 Morse points",
        potential,
        {x: a, y: a, z: a},
        a,
        a**4 - 1,
        4 * a,
    )


def validate_Q() -> None:
    potential = x + y + z + 1 / (x * y) + 1 / (x * z)
    assert_saturated_ideal(
        "Q critical ideal",
        log_system(potential),
        (x - 2 * z, y - z, 2 * z**3 - 1),
    )
    check_parametric_points(
        "Q Morse points",
        potential,
        {x: 2 * b, y: b, z: b},
        b,
        2 * b**3 - 1,
        6 * b,
    )


def validate_B5() -> None:
    potential = x + y + z + 1 / x + 1 / y + 1 / z + 1 / (x * y * z)
    assert_saturated_ideal(
        "B5 critical ideal",
        log_system(potential),
        (x - y, y - z, z**4 - z**2 - 1),
    )
    check_parametric_points(
        "B5 Morse points",
        potential,
        {x: a, y: a, z: a},
        a,
        a**4 - a**2 - 1,
        3 * a + 3 / a + 1 / a**3,
        expected_log_hessian_det=(a**2 + 1) ** 2 * (a**4 + a**2 + 3) / a**5,
    )


def validate_B4() -> None:
    potential = (
        x
        + y * z**2
        + 2 * y * z
        + y
        + 2 / (y * z)
        + 1 / (x * y**2 * z**2)
    )
    assert_saturated_ideal(
        "B4 critical ideal",
        log_system(potential),
        (
            x * y - 1,
            (x - y) * (z + 1),
            (y - 1) * (y + 1) * (z + 1),
            (z - 1) * (z + 1),
        ),
    )
    check_exact_point(
        "B4 positive Morse point",
        potential,
        {x: 1, y: 1, z: 1},
        8,
        expected_log_hessian_det=32,
    )
    check_exact_point(
        "B4 negative Morse point",
        potential,
        {x: -1, y: -1, z: 1},
        -8,
        expected_log_hessian_det=-32,
    )
    check_branch("B4 Morse--Bott curve", potential, {x: t, y: 1 / t, z: -1}, 0, (t,))

    substituted = sp.factor(
        potential.subs({x: t, y: (1 + U) / t, z: V - 1})
    )
    epsilon = sp.symbols("epsilon_B4")
    expansion = sp.series(
        substituted.subs({U: epsilon * U, V: epsilon * V}),
        epsilon,
        0,
        3,
    ).removeO()
    quadratic = sp.factor(sp.expand(expansion).coeff(epsilon, 2))
    expected = t * (U - V) ** 2 + V**2 / t
    assert_equal("B4 transverse quadratic form", quadratic, expected)
    assert_equal(
        "B4 global hyperbolic splitting",
        expected,
        (t * (U - V) + sp.I * V) * ((U - V) - sp.I * V / t),
    )


def validate_B3() -> None:
    potential = (
        x
        + y
        + z
        + 2 / x
        + 2 / y
        + 2 / z
        + x / (y * z)
        + y / (x * z)
        + z / (x * y)
    )
    assert_saturated_ideal(
        "B3 critical ideal",
        log_system(potential),
        (x - y, y - z, z**2 - 3),
    )
    check_parametric_points(
        "B3 Morse points",
        potential,
        {x: a, y: a, z: a},
        a,
        a**2 - 3,
        6 * a,
        expected_log_hessian_det=54 * a,
    )


def validate_B2() -> None:
    expanded = x * (y + z + 1) ** 2 + 2 * (y + z + 1) / (x * y * z) + 1 / (x**3 * y**2 * z**2)
    residual = y + z + 1 + 1 / (x**2 * y * z)
    potential = x * residual**2
    assert_equal("B2 compact factorization", expanded, potential)
    assert_equal(
        "B2 global normal derivative",
        sp.diff(residual, x) * (-x**3 * y * z / 2),
        1,
    )
    complement = divide_log_system("B2", potential, residual)
    assert_saturated_ideal(
        "B2 isolated complement",
        complement,
        (x**2 - 1, y - 1, z - 1),
    )
    check_exact_point(
        "B2 positive Morse point",
        potential,
        {x: 1, y: 1, z: 1},
        16,
        expected_log_hessian_det=512,
    )
    check_exact_point(
        "B2 negative Morse point",
        potential,
        {x: -1, y: 1, z: 1},
        -16,
        expected_log_hessian_det=-512,
    )
    assert_equal(
        "B2 twist-detecting curve",
        residual.subs({x: t, y: -1, z: 1 / t}),
    )


def validate_B1() -> None:
    A = x + y
    q = 1 / (x * y * z**4)
    G = (z + 1) ** 2 + q
    potential = A * G**2
    expanded = (
        x * z**4
        + 4 * x * z**3
        + 6 * x * z**2
        + 4 * x * z
        + x
        + y * z**4
        + 4 * y * z**3
        + 6 * y * z**2
        + 4 * y * z
        + y
        + 2 / (y * z**2)
        + 4 / (y * z**3)
        + 2 / (y * z**4)
        + 2 / (x * z**2)
        + 4 / (x * z**3)
        + 2 / (x * z**4)
        + 1 / (x * y**2 * z**8)
        + 1 / (x**2 * y * z**8)
    )
    assert_equal("B1 expanded potential", expanded, potential)
    expected_logs = (
        G * (x * G - 2 * A * q),
        G * (y * G - 2 * A * q),
        4 * A * G * (z * (z + 1) - 2 * q),
    )
    for variable, actual, expected in zip(VARS, log_system(potential), expected_logs):
        assert_equal(f"B1 theta_{variable} factorization", actual, expected)
    assert_equal("B1 surface smoothness", sp.diff(G, x) * (-x**2 * y * z**4), 1)
    assert_equal(
        "B1 twist-detecting curve",
        G.subs({y: -1 / (4 * x), z: 1}),
    )
    assert_equal(
        "B1 restricted coefficient",
        A.subs({y: -1 / (4 * x), z: 1}),
        (4 * x**2 - 1) / (4 * x),
    )
    complement = divide_log_system("B1", potential, G)
    assert_saturated_ideal(
        "B1 isolated complement",
        complement,
        (x - y, z - 2, 48 * y**2 - 1),
        x * y * z * (x * y * z**4 * (z + 1) ** 2 + 1),
    )
    check_parametric_points(
        "B1 Morse points",
        potential,
        {x: a, y: a, z: 2},
        a,
        48 * a**2 - 1,
        288 * a,
        expected_log_hessian_det=124416 * a,
    )


def validate_V22() -> None:
    potential = (
        x * y
        + x * y / z
        + x
        + y
        + 2 * y / z
        + z
        + 2 * z / y
        + 1 / y
        + z / y**2
        + y / (x * z)
        + 1 / x
    )
    assert_saturated_ideal(
        "V22 critical ideal",
        log_system(potential),
        (x - z, y - z, z**3 + 2 * z**2 - 2),
    )
    check_parametric_points(
        "V22 Morse points",
        potential,
        {x: a, y: a, z: a},
        a,
        a**3 + 2 * a**2 - 2,
        (a**3 + 4 * a**2 + 4 * a + 4) / a,
        expected_log_hessian_det=4 * (a + 1) ** 4 * (a**3 + a**2 + 1) / a**3,
    )


def validate_V18() -> None:
    potential = (
        x * y**2
        + 2 * x * y
        + x
        + 2 * y
        + z
        + 1 / z
        + 2 / y
        + 1 / (y * z)
        + 1 / x
        + 2 * z / (x * y)
        + 1 / (x * y)
        + 1 / (x * y**2)
        + z / (x**2 * y**2)
    )
    assert_saturated_ideal(
        "V18 critical ideal",
        log_system(potential),
        (2 * x + 12 * z - 11, 3 * y - 6 * z + 1, 36 * z**2 - 12 * z - 11),
    )
    check_parametric_points(
        "V18 Morse points",
        potential,
        {x: (11 - 12 * a) / 2, y: (6 * a - 1) / 3, z: a},
        a,
        36 * a**2 - 12 * a - 11,
        18 * a + 3,
        expected_log_hessian_det=9 * (42 * a + 17),
    )


def validate_V16() -> None:
    potential = (
        x
        + 2 * x / (y * z)
        + x / (y**2 * z**2)
        + y * z**2
        + 2 * y * z
        + y
        + 2 * z
        + 2 / z
        + 1 / y
        + 2 / (y * z)
        + 1 / (y * z**2)
        + z / x
        + 2 / x
        + 1 / (x * z)
    )
    check_branch("V16 Morse--Bott curve", potential, {x: t, y: 1, z: -1}, -4, (t,))
    check_parametric_points(
        "V16 Morse points",
        potential,
        {x: 4 - 2 * a, y: a, z: 1},
        a,
        a**2 - 2,
        8 * a + 8,
        expected_log_hessian_det=16 * (17 * a + 24),
    )
    substituted = potential.subs({x: t, y: 1 + U, z: -1 + V}) + 4
    expected = t * U**2 - 2 * t * U * V + (t**2 - 1) * V**2 / t
    check_expansion("V16 transverse quadratic expansion", substituted, expected)


def validate_V14() -> None:
    potential = (
        x**2 / (y * z**2)
        + x
        + 2 * x / z
        + 4 * x / (y * z)
        + y
        + z
        + 2 / z
        + 6 / y
        + 2 * y / x
        + 2 * z / x
        + 4 / x
        + 4 * z / (x * y)
        + y / x**2
        + 2 * z / x**2
        + z**2 / (x**2 * y)
    )
    assert_saturated_ideal(
        "V14 critical ideal",
        log_system(potential),
        (x - 3, 80 * y - 243, 5 * z - 12),
    )
    check_exact_point(
        "V14 Morse point",
        potential,
        {x: 3, y: sp.Rational(243, 80), z: sp.Rational(12, 5)},
        23,
    )


def validate_V12() -> None:
    potential = (
        x**2 * y**3 * z
        + x**2 * y**2 * z
        + 2 * x * y**2 * z
        + x * y**2
        + 2 * x * y * z
        + 2 * x * y
        + x
        + y * z
        + 3 * y
        + z
        + 2 / y
        + 1 / x
        + 1 / (x * z)
        + 2 / (x * y)
        + 3 / (x * y * z)
        + 1 / (x * y**2)
        + 3 / (x * y**2 * z)
        + 1 / (x * y**3 * z)
    )
    check_branch(
        "V12 Morse--Bott curve",
        potential,
        {x: t, y: -1, z: -1 / (t - 1) ** 2},
        -5,
        (t, t - 1),
    )
    check_parametric_points(
        "V12 Morse points",
        potential,
        {x: 2 * a - 1, y: (16 * a - 8) / 9, z: a},
        a,
        32 * a**2 - 32 * a - 1,
        32 * a - 4,
        expected_log_hessian_det=8 * (1154 * a + 35) / 3,
    )
    substituted = potential.subs(
        {x: t, y: -1 + U, z: (V - 1) / (t - 1) ** 2}
    ) + 5
    expected = (t**3 - t**2 + 3 * t - 1) * U**2 / (t * (t - 1)) + U * V
    check_expansion("V12 transverse quadratic expansion", substituted, expected)


def validate_V10() -> None:
    potential = (
        x * y**3 / z
        + x * y**2
        + 4 * x * y**2 / z
        + 2 * x * y
        + 6 * x * y / z
        + x
        + 4 * x / z
        + x / (y * z)
        + 3 * y
        + z
        + 3 / y
        + 2 / x
        + 3 * z / (x * y)
        + 4 / (x * y)
        + 2 / (x * y**2)
        + 3 * z / (x**2 * y**2)
        + z / (x**3 * y**3)
    )
    residual = x * y + 1
    vertical = y + 1
    F1 = x**2 * y * vertical**2 + z * residual
    F2 = x**2 * y * vertical**2 + z * residual**2
    assert_equal("V10 exact D4 factorization", potential + 6, F1 * F2 / (x**3 * y**3 * z))
    check_branch("V10 residual curve", potential, {x: 1, y: -1, z: t}, -6, (t,))
    check_parametric_points(
        "V10 Morse points",
        potential,
        {x: (a + 4) / 4, y: 1, z: a},
        a,
        a**2 + 4 * a - 16,
        5 * a + 26,
        expected_log_hessian_det=8 * (123 * a + 796),
    )


def validate_V8() -> None:
    A = 1 + 1 / (x * y * z)
    B = z + 1
    residual = y * A**2 + B**2
    potential = x * residual**2 - 8
    expanded = (
        x * y**2
        + 2 * x * y * z**2
        + 4 * x * y * z
        + 2 * x * y
        + x * z**4
        + 4 * x * z**3
        + 6 * x * z**2
        + 4 * x * z
        + x
        + 4 * y / z
        + 4 * z
        + 4 / z
        + 6 / (x * z**2)
        + 2 / (x * y)
        + 4 / (x * y * z)
        + 2 / (x * y * z**2)
        + 4 / (x**2 * y * z**3)
        + 1 / (x**3 * y**2 * z**4)
    )
    assert_equal("V8 compact factorization", expanded, potential)
    complement = divide_log_system("V8", potential, residual)
    residual_numerator = (x * y * z + 1) ** 2 + x**2 * y * z**2 * (z + 1) ** 2
    assert_saturated_ideal(
        "V8 isolated complement",
        complement,
        (x - 1, y - 1, z - 1),
        x * y * z * residual_numerator,
    )
    check_exact_point(
        "V8 Morse point",
        potential,
        {x: 1, y: 1, z: 1},
        56,
        expected_ordinary_hessian_det=32768,
    )
    check_branch("V8 crossing curve", potential, {x: t, y: 1 / t, z: -1}, -8, (t,))
    normal_minor = sp.Matrix([A, B]).jacobian((y, z)).det()
    assert_equal(
        "V8 global formal-coordinate minor",
        normal_minor.subs({x: t, y: 1 / t, z: -1}),
        t,
    )


def validate_V6() -> None:
    q = 1 / (x * y * z)
    residual = y + z * (1 + q) ** 2 + 1
    potential = x * residual**3 - 12
    expanded = (
        x * y**3
        + 3 * x * y**2 * z
        + 3 * x * y**2
        + 3 * x * y * z**2
        + 6 * x * y * z
        + 3 * x * y
        + x * z**3
        + 3 * x * z**2
        + 3 * x * z
        + x
        + 6 * y
        + 12 * z
        + 6 * z**2 / y
        + 12 * z / y
        + 6 / y
        + 3 / (x * z)
        + 18 / (x * y)
        + 6 / (x * y * z)
        + 15 * z / (x * y**2)
        + 18 / (x * y**2)
        + 3 / (x * y**2 * z)
        + 12 / (x**2 * y**2 * z)
        + 20 / (x**2 * y**3)
        + 12 / (x**2 * y**3 * z)
        + 3 / (x**3 * y**3 * z**2)
        + 15 / (x**3 * y**4 * z)
        + 3 / (x**3 * y**4 * z**2)
        + 6 / (x**4 * y**5 * z**2)
        + 1 / (x**5 * y**6 * z**3)
    )
    assert_equal("V6 compact factorization", expanded, potential)
    assert_equal(
        "V6 x-derivative formula",
        sp.diff(residual, x),
        -2 * z * q * (1 + q) / x,
    )
    assert_equal(
        "V6 y-derivative formula",
        sp.diff(residual, y),
        1 - 2 * z * q * (1 + q) / y,
    )
    complement = divide_log_system("V6", potential + 12, residual**2)
    assert_saturated_ideal(
        "V6 isolated complement", complement, (x - 4, y - 1, 4 * z - 1)
    )
    check_exact_point("V6 Morse point", potential, {x: 4, y: 1, z: sp.Rational(1, 4)}, 96)


def validate_V4() -> None:
    residual = y + z + 1 + 1 / (x * y * z)
    potential = x * residual**4 - 24
    expanded = (
        x * y**4
        + 4 * x * y**3 * z
        + 4 * x * y**3
        + 6 * x * y**2 * z**2
        + 12 * x * y**2 * z
        + 6 * x * y**2
        + 4 * x * y * z**3
        + 12 * x * y * z**2
        + 12 * x * y * z
        + 4 * x * y
        + x * z**4
        + 4 * x * z**3
        + 6 * x * z**2
        + 4 * x * z
        + x
        + 4 * y**2 / z
        + 12 * y
        + 12 * y / z
        + 12 * z
        + 12 / z
        + 4 * z**2 / y
        + 12 * z / y
        + 12 / y
        + 4 / (y * z)
        + 6 / (x * z**2)
        + 12 / (x * y * z)
        + 12 / (x * y * z**2)
        + 6 / (x * y**2)
        + 12 / (x * y**2 * z)
        + 6 / (x * y**2 * z**2)
        + 4 / (x**2 * y**2 * z**3)
        + 4 / (x**2 * y**3 * z**2)
        + 4 / (x**2 * y**3 * z**3)
        + 1 / (x**3 * y**4 * z**4)
    )
    assert_equal("V4 compact factorization", expanded, potential)
    assert_equal(
        "V4 global normal derivative",
        sp.diff(residual, x) * (-x**2 * y * z),
        1,
    )
    complement = divide_log_system("V4", potential + 24, residual**3)
    assert_saturated_ideal(
        "V4 isolated complement", complement, (x - 1, y - 1, z - 1)
    )
    check_exact_point("V4 Morse point", potential, {x: 1, y: 1, z: 1}, 232)


def validate_V2() -> None:
    q = 1 / (x * y**3 * z)
    residual = y + z + 1 + q
    potential = x * residual**6 - 120
    expected_logs = (
        x * residual**5 * (residual - 6 * q),
        6 * x * residual**5 * (y - 3 * q),
        6 * x * residual**5 * (z - q),
    )
    for variable, actual, expected in zip(VARS, log_system(potential), expected_logs):
        assert_equal(f"V2 theta_{variable} factorization", actual, expected)
    assert_equal(
        "V2 global normal derivative",
        sp.diff(residual, x) * (-x**2 * y**3 * z),
        1,
    )
    complement = divide_log_system("V2", potential + 120, residual**5)
    assert_saturated_ideal(
        "V2 isolated complement", complement, (27 * x - 1, y - 3, z - 1)
    )
    expected_matrix = sp.Matrix(
        [[1440, 5184, 1728], [5184, 20736, 5184], [1728, 5184, 3456]]
    )
    check_exact_point(
        "V2 Morse point",
        potential,
        {x: sp.Rational(1, 27), y: 3, z: 1},
        1608,
        expected_log_hessian_det=2**17 * 3**9,
        expected_log_hessian_matrix=expected_matrix,
    )


CASE_ORDER = (
    "P3",
    "Q",
    "B_5",
    "B_4",
    "B_3",
    "B_2",
    "B_1",
    "V_22",
    "V_18",
    "V_16",
    "V_14",
    "V_12",
    "V_10",
    "V_8",
    "V_6",
    "V_4",
    "V_2",
)

CASES: dict[str, Callable[[], None]] = {
    "P3": validate_P3,
    "Q": validate_Q,
    "B_5": validate_B5,
    "B_4": validate_B4,
    "B_3": validate_B3,
    "B_2": validate_B2,
    "B_1": validate_B1,
    "V_22": validate_V22,
    "V_18": validate_V18,
    "V_16": validate_V16,
    "V_14": validate_V14,
    "V_12": validate_V12,
    "V_10": validate_V10,
    "V_8": validate_V8,
    "V_6": validate_V6,
    "V_4": validate_V4,
    "V_2": validate_V2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Picard-rank-one Fano threefold computations."
    )
    parser.add_argument("--list", action="store_true", help="List available cases.")
    parser.add_argument(
        "--coverage", action="store_true", help="Report expected and implemented cases."
    )
    parser.add_argument(
        "--case", action="append", choices=CASE_ORDER, help="Case to run; repeatable."
    )
    parser.add_argument("--all", action="store_true", help="Run all 17 cases.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for name in CASE_ORDER:
            print(name)
        return 0
    if args.coverage:
        print(f"Expected Picard-rank-one cases: {len(CASE_ORDER)}")
        print(f"Implemented cases: {len(CASES)}")
        missing = [name for name in CASE_ORDER if name not in CASES]
        print(f"Missing cases: {len(missing)}")
        for name in missing:
            print(f"  {name}")
        return 0

    selected = CASE_ORDER if args.all else tuple(args.case or ())
    if not selected:
        print("No case selected. Use --list, --case NAME, or --all.")
        return 2

    for name in selected:
        print(f"\n== {name} ==")
        CASES[name]()
        print("PASS")

    print(f"\nAll {len(selected)} selected Picard-rank-one checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())