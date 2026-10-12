#!/usr/bin/env python3
"""
Reproducible symbolic checks for selected Mori--Mukai Landau--Ginzburg examples.

The script uses only SymPy.  It checks the ingredients that appear repeatedly in
Chapter 4 of the manuscript:

  * logarithmic critical systems theta_i(W)=x_i dW/dx_i;
  * clearing denominators and saturation by the torus monomial xyz;
  * ordinary and logarithmic Hessian determinants at critical points;
  * exact unit-times-power normal forms such as W-c = unit * F^n;
  * parametric branches modulo a minimal polynomial.

Run examples:

  python computations/mori_mukai_validation.py --list
  python computations/mori_mukai_validation.py --case MM_2_10
  python computations/mori_mukai_validation.py --all
  python computations/mori_mukai_validation.py --all --verbose

The cases below are intentionally written as ordinary Python/SymPy expressions,
not parsed from LaTeX.  This makes the verification independent of typesetting
changes in main.tex.  Every label in ``CASES`` has assertive checks: saturated
critical ideals, parametrized branches, Hessians, or normal-form identities,
depending on the geometry of the example.  The ``--coverage`` option reports
Mori--Mukai labels in the manuscript that are not yet represented in ``CASES``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

x, y, z = sp.symbols("x y z")
t = sp.symbols("t")
U, V = sp.symbols("U V")
alpha, beta, eta, lam, rho, xi, a = sp.symbols("alpha beta eta lam rho xi a")
TORUS_VARS = (x, y, z)


@dataclass(frozen=True)
class PointCheck:
    label: str
    point: Dict[sp.Symbol, sp.Expr]
    value: Optional[sp.Expr] = None
    ordinary_hessian_det: Optional[sp.Expr] = None
    log_hessian_det: Optional[sp.Expr] = None
    require_nonzero_ordinary: bool = False
    require_nonzero_log: bool = False


@dataclass(frozen=True)
class ParametricPointCheck:
    label: str
    parameter: sp.Symbol
    minimal_polynomial: sp.Expr
    point: Dict[sp.Symbol, sp.Expr]
    value: Optional[sp.Expr] = None
    ordinary_hessian_det: Optional[sp.Expr] = None
    log_hessian_det: Optional[sp.Expr] = None
    coprime_tests: Tuple[sp.Expr, ...] = ()


@dataclass(frozen=True)
class SaturationCheck:
    label: str
    expected_generators: Tuple[sp.Expr, ...]
    by: sp.Expr = x * y * z


@dataclass(frozen=True)
class NormalFormCheck:
    label: str
    critical_value: sp.Expr
    residual: sp.Expr
    exponent: int
    unit: sp.Expr
    extra_term: sp.Expr = sp.Integer(0)


@dataclass(frozen=True)
class ExpansionCheck:
    label: str
    substitution: Dict[sp.Symbol, sp.Expr]
    expected: sp.Expr
    normal_variables: Tuple[sp.Symbol, ...]
    order: int = 3


@dataclass(frozen=True)
class ParametricBranchCheck:
    label: str
    parameter: sp.Symbol
    substitution: Dict[sp.Symbol, sp.Expr]
    exclusions: Tuple[sp.Expr, ...] = ()
    value: Optional[sp.Expr] = None
    log_hessian_matrix: Optional[sp.Matrix] = None


@dataclass(frozen=True)
class Case:
    name: str
    potential: sp.Expr
    variables: Tuple[sp.Symbol, ...] = TORUS_VARS
    notes: str = ""
    identities: Tuple[Tuple[str, sp.Expr], ...] = ()
    saturation_checks: Tuple[SaturationCheck, ...] = ()
    point_checks: Tuple[PointCheck, ...] = ()
    parametric_point_checks: Tuple[ParametricPointCheck, ...] = ()
    normal_form_checks: Tuple[NormalFormCheck, ...] = ()
    expansion_checks: Tuple[ExpansionCheck, ...] = ()
    branch_checks: Tuple[ParametricBranchCheck, ...] = ()


class CheckError(AssertionError):
    pass


def theta(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
    """Logarithmic derivative theta_var(expr)=var*d(expr)/dvar."""
    return sp.factor(var * sp.diff(expr, var))


def log_system(W: sp.Expr, variables: Sequence[sp.Symbol]) -> List[sp.Expr]:
    return [theta(W, var) for var in variables]


def cleared_numerators(expressions: Iterable[sp.Expr]) -> List[sp.Expr]:
    """Return numerators after putting Laurent expressions over a common denominator."""
    result = []
    for expr in expressions:
        numerator, _denominator = sp.fraction(sp.together(expr))
        result.append(sp.factor(numerator))
    return result


def log_hessian(W: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Matrix:
    return sp.Matrix([[theta(theta(W, vj), vi) for vj in variables] for vi in variables])


def ordinary_hessian(W: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Matrix:
    return sp.hessian(W, variables)


def saturated_elimination_generators(
    generators: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    saturation_element: sp.Expr,
) -> List[sp.Expr]:
    """
    Compute I : saturation_element^infty using the elimination trick
    I : f^infty = (I, 1 - T*f) \\cap k[variables].

    This is suitable for the zero-dimensional torus critical ideals appearing in
    the examples.  For very large examples a CAS such as Sage/Singular may be
    faster, but the output here is reproducible with the packages preinstalled in
    this workspace.
    """
    aux = sp.symbols("T_sat")
    groebner_basis = sp.groebner(
        [sp.factor(g) for g in generators] + [1 - aux * saturation_element],
        aux,
        *variables,
        order="lex",
    )
    eliminated = [g.as_expr() for g in groebner_basis.polys if not g.as_expr().has(aux)]
    return [sp.factor(g) for g in eliminated]


def groebner_basis(expressions: Sequence[sp.Expr], variables: Sequence[sp.Symbol]) -> sp.GroebnerBasis:
    return sp.groebner([sp.factor(e) for e in expressions], *variables, order="lex")


def ideals_equal(
    left: Sequence[sp.Expr], right: Sequence[sp.Expr], variables: Sequence[sp.Symbol]
) -> bool:
    """Check equality by mutual reduction of generators."""
    G_left = groebner_basis(left, variables)
    G_right = groebner_basis(right, variables)
    left_in_right = all(sp.factor(G_right.reduce(g)[1]) == 0 for g in left)
    right_in_left = all(sp.factor(G_left.reduce(g)[1]) == 0 for g in right)
    return left_in_right and right_in_left


def rational_mod_equal(expr: sp.Expr, expected: sp.Expr, modulus: sp.Expr, parameter: sp.Symbol) -> bool:
    """Check equality in Q(parameter)/(modulus) for rational expressions."""
    numerator, denominator = sp.fraction(sp.together(expr - expected))
    remainder = sp.rem(sp.Poly(numerator, parameter), sp.Poly(modulus, parameter)).as_expr()
    if sp.factor(remainder) != 0:
        return False
    denominator_remainder = sp.rem(
        sp.Poly(sp.fraction(sp.together(denominator))[0], parameter),
        sp.Poly(modulus, parameter),
    ).as_expr()
    return sp.gcd(sp.Poly(denominator_remainder, parameter), sp.Poly(modulus, parameter)).degree() == 0


def coprime_mod(expr: sp.Expr, modulus: sp.Expr, parameter: sp.Symbol) -> bool:
    numerator, denominator = sp.fraction(sp.together(expr))
    num_rem = sp.rem(sp.Poly(numerator, parameter), sp.Poly(modulus, parameter)).as_expr()
    den_rem = sp.rem(sp.Poly(denominator, parameter), sp.Poly(modulus, parameter)).as_expr()
    return (
        sp.gcd(sp.Poly(num_rem, parameter), sp.Poly(modulus, parameter)).degree() == 0
        and sp.gcd(sp.Poly(den_rem, parameter), sp.Poly(modulus, parameter)).degree() == 0
    )


def total_degree_truncation(expr: sp.Expr, normal_variables: Sequence[sp.Symbol], order: int) -> sp.Expr:
    """Return terms of total normal degree < order using a scaling variable."""
    scale = sp.symbols("eps_total_degree")
    scaled = sp.together(expr.subs({var: scale * var for var in normal_variables}))
    truncated = sp.series(scaled, scale, 0, order).removeO()
    return sp.factor(sp.together(truncated.subs(scale, 1)))


def assert_equal(label: str, expr: sp.Expr, expected: sp.Expr) -> None:
    difference = sp.factor(sp.together(expr - expected))
    if difference != 0:
        raise CheckError(f"{label}: expected {expected}, got {expr}; difference {difference}")


def assert_nonzero(label: str, expr: sp.Expr) -> None:
    if sp.simplify(expr) == 0:
        raise CheckError(f"{label}: expression unexpectedly vanished")


def case_has_assertive_checks(case: Case) -> bool:
    return any(
        (
            case.identities,
            case.saturation_checks,
            case.point_checks,
            case.parametric_point_checks,
            case.normal_form_checks,
            case.expansion_checks,
            case.branch_checks,
        )
    )


def needs_ordinary_hessian(case: Case) -> bool:
    return any(
        point.ordinary_hessian_det is not None or point.require_nonzero_ordinary
        for point in case.point_checks
    ) or any(point.ordinary_hessian_det is not None for point in case.parametric_point_checks)


def needs_log_hessian(case: Case) -> bool:
    return (
        any(point.log_hessian_det is not None or point.require_nonzero_log for point in case.point_checks)
        or any(point.log_hessian_det is not None for point in case.parametric_point_checks)
        or any(branch.log_hessian_matrix is not None for branch in case.branch_checks)
    )


def check_case(case: Case, verbose: bool = False) -> None:
    print(f"\n== {case.name} ==")
    if case.notes:
        print(case.notes)

    W = case.potential
    variables = case.variables
    logs = log_system(W, variables)
    cleared = cleared_numerators(logs)
    print("logarithmic system: computed")
    if verbose:
        for var, eq, num in zip(variables, logs, cleared):
            print(f"  theta_{var} W = {sp.factor(eq)}")
            print(f"  cleared numerator = {num}")

    for label, identity in case.identities:
        assert_equal(label, identity, 0)
        print(f"identity: {label}: ok")

    for saturation in case.saturation_checks:
        computed = saturated_elimination_generators(cleared, variables, saturation.by)
        if verbose:
            print(f"saturation {saturation.label}: computed generators")
            for gen in computed:
                print(f"  {gen}")
        if not ideals_equal(computed, saturation.expected_generators, variables):
            raise CheckError(
                f"{case.name}/{saturation.label}: saturated ideal differs from expected generators\n"
                f"computed={computed}\nexpected={list(saturation.expected_generators)}"
            )
        print(f"saturation: {saturation.label}: ok")

    H_ord = ordinary_hessian(W, variables) if needs_ordinary_hessian(case) else None
    H_log = log_hessian(W, variables) if needs_log_hessian(case) else None

    if not case_has_assertive_checks(case):
        print("baseline entry: potential and logarithmic system only")

    for point in case.point_checks:
        subs = point.point
        for equation in logs:
            assert_equal(
                f"{point.label} logarithmic critical equation",
                sp.simplify(equation.subs(subs)),
                0,
            )
        if point.value is not None:
            assert_equal(f"{point.label} value", sp.simplify(W.subs(subs)), point.value)
        if point.ordinary_hessian_det is not None or point.require_nonzero_ordinary:
            assert H_ord is not None
            det_ord = sp.factor(H_ord.det().subs(subs))
            if point.ordinary_hessian_det is not None:
                assert_equal(f"{point.label} ordinary Hessian det", det_ord, point.ordinary_hessian_det)
            if point.require_nonzero_ordinary:
                assert_nonzero(f"{point.label} ordinary Hessian det", det_ord)
            if verbose:
                print(f"  {point.label}: det ordinary H = {det_ord}")
        if point.log_hessian_det is not None or point.require_nonzero_log:
            assert H_log is not None
            det_log = sp.factor(H_log.det().subs(subs))
            if point.log_hessian_det is not None:
                assert_equal(f"{point.label} log Hessian det", det_log, point.log_hessian_det)
            if point.require_nonzero_log:
                assert_nonzero(f"{point.label} log Hessian det", det_log)
            if verbose:
                print(f"  {point.label}: det log H = {det_log}")
        print(f"point: {point.label}: ok")

    for point in case.parametric_point_checks:
        subs = point.point
        parameter = point.parameter
        modulus = point.minimal_polynomial
        for equation in logs:
            if not rational_mod_equal(
                sp.simplify(equation.subs(subs)), 0, modulus, parameter
            ):
                raise CheckError(
                    f"{point.label}: logarithmic critical equation does not vanish "
                    f"modulo {modulus}"
                )
        if point.value is not None:
            if not rational_mod_equal(sp.simplify(W.subs(subs)), point.value, modulus, parameter):
                raise CheckError(f"{point.label}: critical value mismatch modulo {modulus}")
        if point.ordinary_hessian_det is not None:
            assert H_ord is not None
            det_ord = sp.factor(H_ord.det().subs(subs))
            if not rational_mod_equal(det_ord, point.ordinary_hessian_det, modulus, parameter):
                raise CheckError(f"{point.label}: ordinary Hessian mismatch modulo {modulus}")
        if point.log_hessian_det is not None:
            assert H_log is not None
            det_log = sp.factor(H_log.det().subs(subs))
            if not rational_mod_equal(det_log, point.log_hessian_det, modulus, parameter):
                raise CheckError(f"{point.label}: log Hessian mismatch modulo {modulus}")
        for expr in point.coprime_tests:
            if not coprime_mod(expr, modulus, parameter):
                raise CheckError(f"{point.label}: expression {expr} is not coprime to {modulus}")
        print(f"parametric point branch: {point.label}: ok")

    for normal in case.normal_form_checks:
        lhs = sp.simplify(W - normal.critical_value)
        rhs = sp.simplify(normal.extra_term + normal.unit * normal.residual**normal.exponent)
        assert_equal(normal.label, lhs, rhs)
        print(f"normal form: {normal.label}: ok")

    for expansion in case.expansion_checks:
        actual = sp.together((W.subs(expansion.substitution) - expansion.expected))
        low_terms = total_degree_truncation(actual, expansion.normal_variables, expansion.order)
        if low_terms != 0:
            raise CheckError(f"{expansion.label}: expansion has terms below order {expansion.order}: {low_terms}")
        print(f"formal expansion: {expansion.label}: ok modulo degree {expansion.order}")

    for branch in case.branch_checks:
        subs = branch.substitution
        for exclusion in branch.exclusions:
            if sp.simplify(exclusion.subs(subs)) == 0:
                raise CheckError(f"{branch.label}: exclusion {exclusion} vanishes identically")
        for eq in logs:
            if sp.factor(sp.together(eq.subs(subs))) != 0:
                raise CheckError(f"{branch.label}: logarithmic equation does not vanish: {eq.subs(subs)}")
        if branch.value is not None:
            assert_equal(f"{branch.label} value", sp.simplify(W.subs(subs)), branch.value)
        if branch.log_hessian_matrix is not None:
            actual = sp.simplify(H_log.subs(subs))
            if actual != branch.log_hessian_matrix:
                raise CheckError(f"{branch.label}: log Hessian matrix mismatch\n{actual}\n!=\n{branch.log_hessian_matrix}")
        print(f"parametric branch: {branch.label}: ok")


def build_cases() -> Dict[str, Case]:
    sqrt5 = sp.sqrt(5)
    sqrt13 = sp.sqrt(13)

    p_201 = x * y
    F_201 = p_201**2 * z**5 + p_201 * z**2 + 1
    R_201 = F_201**3 / (p_201**2 * z**6)
    W_201 = R_201 * (R_201 + x + y) / p_201 - 60

    F_202 = x * y * (y + z + 1) + 1
    Q_202 = (
        x**3 * y**4 * z
        + x**2 * y**4
        + 2 * x**2 * y**3 * z
        + 2 * x**2 * y**3
        + x**2 * y**2 * z**2
        + 2 * x**2 * y**2 * z
        + x**2 * y**2
        + 2 * x * y**2
        + 2 * x * y * z
        + 2 * x * y
        + 1
    ) / (x**4 * y**6 * z)
    W_202 = F_202**2 * Q_202 - 14
    q_202 = 4 * t**3 - 16 * t**2 + 16 * t - 1

    F_203 = x * y * z * (y + 1) + 1
    Q_203 = (
        x**2 * y**4 * z**2
        + 2 * x**2 * y**3 * z**2
        + x**2 * y**2 * z**2
        + x * y**2 * z**2
        + 2 * x * y**2 * z
        + 2 * x * y * z
        + x * y
        + 1
    ) / (x**2 * y**3 * z**2)
    W_203 = F_203**2 * Q_203 - 12
    A_203 = y * z**2 - 1
    H_203 = x**2 * (z**2 + 1) ** 2 + x * z**3 * (2 * z**2 + z + 2) + z**6

    v_204 = y + 1 + 1 / (y * z)
    W_204 = x * v_204**2 + v_204**2 / (x * y**2) + z * v_204**3 + v_204**2 / y - v_204 / y - 7

    F_208 = x * (y + z + 1) + 1 + 1 / (y * z)
    W_208 = F_208**2 / x - 2

    F_210 = y * z + y + y / x + 2 + 1 / (y * z)
    W_210 = (
        x
        + y * z**2
        + 2 * y * z
        + y
        + 4 * z
        + 2 * y * z / x
        + 2 * y / x
        + 4 / x
        + y / x**2
        + 6 / y
        + 2 / (y * z)
        + 4 / (y**2 * z)
        + 2 / (x * y * z)
        + 1 / (y**3 * z**2)
    )

    A_212 = z + 1
    B_212 = 1 + 1 / (x * z)
    F_212 = y * A_212 + B_212
    W_212 = F_212**2 / y + x * B_212**2 - 2

    A_213 = 1 + y / x
    B_213 = 1 + 1 / y
    F_213 = z * A_213 + B_213
    W_213 = F_213**2 / z + x * B_213 + y - 2

    A_216 = 1 + 1 / y
    B_216 = 1 + y / x
    F_216 = z * A_216 + B_216
    W_216 = F_216**2 / z + x + y - 2
    q_beta = beta**3 - 4 * beta - 16

    A_218 = z + 1 + 1 / y + 1 / (x * y * z)
    W_218 = x + y * A_218**2 - 2
    q_eta = 4 * eta**4 + 4 * eta**3 - 1

    W_219 = (x + 1) ** 2 / (y * z) + z * (x + y) ** 2 / x**2 + x + y * (x + 1) / x
    h_rho_219 = rho**4 - rho**2 - 2 * rho - 1

    W_229 = x + x / y + y + z + 2 / x + 1 / (x**2 * z)
    q_alpha = alpha**4 + alpha**3 - 4

    W_230 = x + y + z + z / (x * y) + 2 / (x * y) + 1 / (x * y * z)
    p_lam = lam**5 + lam**4 + lam**3 - 3 * lam**2 + 3 * lam - 1

    A_81 = y * z * (z + 1)
    u_81 = A_81 + 1
    W_81 = x + 1 / x + y * (z + 1) ** 3 + 3 * (z + 1 / z) + 3 * (z + 1) / (y * z**2) + 1 / (y**2 * z**3)

    F_91 = z + 1 + 1 / (x * z)
    q_91 = 1 / (x * z)
    W_91 = x * F_91**4 + y + 1 / y - 12

    F_101 = z + 1 + 1 / (x * z**2)
    q_101 = 1 / (x * z**2)
    W_101 = x * F_101**6 + y + 1 / y - 60

    cases: Dict[str, Case] = {}

    cases["MM_2_1"] = Case(
        name="MM_2_1",
        potential=W_201,
        notes=(
            "Residual A2 surface with a u^3v crossing, one exact hyperbolic "
            "curve, and two isolated Morse points."
        ),
        identities=(
            (
                "second critical fiber product W+61=(R+x)(R+y)/(xy)",
                W_201 + 61 - (R_201 + x) * (R_201 + y) / (x * y),
            ),
        ),
        normal_form_checks=(
            NormalFormCheck(
                label="surface S: exact residual cubic factor",
                critical_value=-60,
                residual=F_201,
                exponent=3,
                unit=(R_201 + x + y) / (p_201**3 * z**6),
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="a^2=1/108 isolated branch",
                parameter=a,
                minimal_polynomial=108 * a**2 - 1,
                point={x: a, y: a, z: 6},
                value=372 + 432 * a,
                log_hessian_det=186624 * (433 * a + 4),
                coprime_tests=(a, 433 * a + 4),
            ),
        ),
    )

    x_surface_202 = -1 / (y * (y + z + 1))
    cases["MM_2_2"] = Case(
        name="MM_2_2",
        potential=W_202,
        notes="Exact quadratic critical surface plus three isolated Morse points.",
        identities=(
            (
                "surface coefficient Q|S=-(y+z+1)/y",
                Q_202.subs(x, x_surface_202) + (y + z + 1) / y,
            ),
        ),
        normal_form_checks=(
            NormalFormCheck(
                label="surface S: W+14=QF^2",
                critical_value=-14,
                residual=F_202,
                exponent=2,
                unit=Q_202,
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="critical surface F=0",
                parameter=y,
                substitution={x: x_surface_202},
                exclusions=(y, z, y + z + 1),
                value=-14,
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="4t^3-16t^2+16t-1 isolated branch",
                parameter=t,
                minimal_polynomial=q_202,
                point={x: 4 * t - 8, y: t, z: 1},
                value=2 * (24 * t**2 - 32 * t - 5),
                log_hessian_det=128 * (1264 * t**2 - 2237 * t + 144),
                coprime_tests=(t * (t - 2), 1264 * t**2 - 2237 * t + 144),
            ),
        ),
    )

    x_surface_203 = -1 / (y * z * (y + 1))
    cases["MM_2_3"] = Case(
        name="MM_2_3",
        potential=W_203,
        notes=(
            "Quadratic surface with a u^2v crossing, a second quadratic curve, "
            "and two isolated Morse points."
        ),
        identities=(
            (
                "surface coefficient restriction",
                Q_203.subs(x, x_surface_203)
                + (y + 1) * (y * z**2 + 1) / (y * z),
            ),
            (
                "Gamma square identity on yz^2=1",
                (W_203 + 13 - H_203**2 / (x**2 * z**8)).subs(y, 1 / z**2),
            ),
        ),
        normal_form_checks=(
            NormalFormCheck(
                label="surface S: W+12=QF^2",
                critical_value=-12,
                residual=F_203,
                exponent=2,
                unit=Q_203,
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="critical surface F=0",
                parameter=y,
                substitution={x: x_surface_203},
                exclusions=(y, z, y + 1),
                value=-12,
            ),
            ParametricBranchCheck(
                label="crossing curve K",
                parameter=t,
                substitution={x: t**3 / (t**2 - 1), y: -1 / t**2, z: t},
                exclusions=(t, t**2 - 1),
                value=-12,
            ),
        ),
        point_checks=(
            PointCheck(
                label="P_+=(1/2,1,1)",
                point={x: sp.Rational(1, 2), y: 1, z: 1},
                value=68,
                log_hessian_det=41472,
            ),
            PointCheck(
                label="P_-=(-1/2,1,-1)",
                point={x: sp.Rational(-1, 2), y: 1, z: -1},
                value=36,
                log_hessian_det=-25088,
            ),
        ),
    )

    mm91_identities: List[Tuple[str, sp.Expr]] = [
        ("theta_x factorization", theta(W_91, x) - x * F_91**3 * (F_91 - 4 * q_91)),
        ("theta_z factorization", theta(W_91, z) - 4 * x * F_91**3 * (z - q_91)),
        ("theta_y factorization", theta(W_91, y) - (y - 1 / y)),
    ]
    mm91_points: List[PointCheck] = []
    mm91_normals: List[NormalFormCheck] = []
    mm91_branches: List[ParametricBranchCheck] = []
    for epsilon in (1, -1):
        D_epsilon = y - epsilon
        branch_substitution = {
            x: -1 / (t * (t + 1)),
            y: epsilon,
            z: t,
        }
        jacobian = sp.Matrix([z, D_epsilon, F_91]).jacobian(TORUS_VARS).det()
        mm91_identities.append(
            (
                f"Gamma_{epsilon} normal-coordinate Jacobian",
                jacobian.subs(branch_substitution) - t * (t + 1) ** 2,
            )
        )
        mm91_normals.append(
            NormalFormCheck(
                label=f"Gamma_{epsilon}: quadratic plus residual A3",
                critical_value=2 * epsilon - 12,
                residual=F_91,
                exponent=4,
                unit=x,
                extra_term=D_epsilon**2 / y,
            )
        )
        mm91_branches.append(
            ParametricBranchCheck(
                label=f"Gamma_{epsilon}",
                parameter=t,
                substitution=branch_substitution,
                exclusions=(t, t + 1),
                value=2 * epsilon - 12,
            )
        )
        mm91_points.append(
            PointCheck(
                label=f"P_{epsilon}=(4,{epsilon},1/2)",
                point={x: 4, y: epsilon, z: sp.Rational(1, 2)},
                value=52 + 2 * epsilon,
                log_hessian_det=4096 * epsilon,
            )
        )
    cases["MM_9_1"] = Case(
        name="MM_9_1",
        potential=W_91,
        notes="Two twisted mixed A3 curves and two isolated Morse points.",
        identities=tuple(mm91_identities),
        point_checks=tuple(mm91_points),
        normal_form_checks=tuple(mm91_normals),
        branch_checks=tuple(mm91_branches),
    )

    mm101_identities: List[Tuple[str, sp.Expr]] = [
        ("theta_x factorization", theta(W_101, x) - x * F_101**5 * (F_101 - 6 * q_101)),
        ("theta_z factorization", theta(W_101, z) - 6 * x * F_101**5 * (z - 2 * q_101)),
        ("theta_y factorization", theta(W_101, y) - (y - 1 / y)),
    ]
    mm101_points: List[PointCheck] = []
    mm101_normals: List[NormalFormCheck] = []
    mm101_branches: List[ParametricBranchCheck] = []
    for epsilon in (1, -1):
        D_epsilon = y - epsilon
        branch_substitution = {
            x: -1 / (t**2 * (t + 1)),
            y: epsilon,
            z: t,
        }
        jacobian = sp.Matrix([z, D_epsilon, F_101]).jacobian(TORUS_VARS).det()
        mm101_identities.append(
            (
                f"Gamma_{epsilon} normal-coordinate Jacobian",
                jacobian.subs(branch_substitution) - t**2 * (t + 1) ** 2,
            )
        )
        mm101_normals.append(
            NormalFormCheck(
                label=f"Gamma_{epsilon}: quadratic plus residual A5",
                critical_value=2 * epsilon - 60,
                residual=F_101,
                exponent=6,
                unit=x,
                extra_term=D_epsilon**2 / y,
            )
        )
        mm101_branches.append(
            ParametricBranchCheck(
                label=f"Gamma_{epsilon}",
                parameter=t,
                substitution=branch_substitution,
                exclusions=(t, t + 1),
                value=2 * epsilon - 60,
            )
        )
        mm101_points.append(
            PointCheck(
                label=f"P_{epsilon}=(27/4,{epsilon},2/3)",
                point={x: sp.Rational(27, 4), y: epsilon, z: sp.Rational(2, 3)},
                value=372 + 2 * epsilon,
                log_hessian_det=373248 * epsilon,
            )
        )
    cases["MM_10_1"] = Case(
        name="MM_10_1",
        potential=W_101,
        notes="Two twisted mixed A5 curves and two isolated Morse points.",
        identities=tuple(mm101_identities),
        point_checks=tuple(mm101_points),
        normal_form_checks=tuple(mm101_normals),
        branch_checks=tuple(mm101_branches),
    )

    mm204_points: List[PointCheck] = []
    for sign in (1, -1):
        y0 = (-1 + sign * sqrt5) / 2
        mm204_points.append(
            PointCheck(
                label=f"xy=-1 branch sign {sign}",
                point={x: -1 / y0, y: y0, z: y0},
                value=(-3 + sign * 5 * sqrt5) / 2,
                require_nonzero_ordinary=True,
            )
        )
    for sign in (1, -1):
        y0 = (-1 + sign * 5 * sqrt13) / 18
        mm204_points.append(
            PointCheck(
                label=f"xy=1 branch sign {sign}",
                point={x: 1 / y0, y: y0, z: (153 * y0 - 64) / 45},
                value=(33 + sign * 13 * sqrt13) / 2,
                require_nonzero_ordinary=True,
            )
        )

    cases["MM_2_4"] = Case(
        name="MM_2_4",
        potential=W_204,
        notes="Four isolated Morse points from the two xy=+-1 branches.",
        point_checks=tuple(mm204_points),
    )

    S_205 = x * y / z + 1 + 1 / (x * y)
    W_205 = y * S_205**2 + x * S_205 + z * S_205**3 + x * y / z - 6
    gamma_205_polynomial = t**3 + t**2 - 1
    A_205 = -2 / (t * gamma_205_polynomial)
    B_205 = t**3 * (4 * t + 3) / gamma_205_polynomial**3
    C_205 = -t**7 * (t + 1) * (2 * t + 1) / gamma_205_polynomial**5

    B_206 = (z + 1) ** 2 / z
    W_206 = x * (1 + 1 / (y * z)) ** 2 + y * z * B_206 + B_206**2 / x + 2 * (B_206 - 2) + 2 * B_206 / (y * z)

    cases["MM_2_8"] = Case(
        name="MM_2_8",
        potential=W_208,
        notes="Exact square surface contribution plus three isolated Morse points.",
        identities=(("compact square W+2=F^2/x", W_208 + 2 - F_208**2 / x),),
        normal_form_checks=(
            NormalFormCheck(
                label="surface Sigma: W+2=(1/x)F^2",
                critical_value=-2,
                residual=F_208,
                exponent=2,
                unit=1 / x,
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="rho^3-rho-1 isolated branch",
                parameter=rho,
                minimal_polynomial=rho**3 - rho - 1,
                point={x: rho ** (-3), y: rho, z: rho},
                ordinary_hessian_det=32 * (51 * rho**2 + 64 * rho + 33) / (rho * (rho + 1)),
                coprime_tests=(51 * rho**2 + 64 * rho + 33, rho * (rho + 1)),
            ),
        ),
    )

    cases["MM_2_5"] = Case(
        name="MM_2_5",
        potential=W_205,
        notes="Positive-dimensional A1 curve plus two isolated Morse points.",
        identities=(
            ("quadratic normal discriminant on Gamma", 4 * A_205 * C_205 - B_205**2 + t**6 / gamma_205_polynomial**6),
        ),
        point_checks=(
            PointCheck(
                label="p_+",
                point={x: sp.sqrt(3), y: (9 + 25 * sp.sqrt(3)) / 78, z: (243 + 77 * sp.sqrt(3)) / 234},
                value=21 + 6 * sp.sqrt(3),
                require_nonzero_ordinary=True,
            ),
            PointCheck(
                label="p_-",
                point={x: -sp.sqrt(3), y: (9 - 25 * sp.sqrt(3)) / 78, z: (243 - 77 * sp.sqrt(3)) / 234},
                value=21 - 6 * sp.sqrt(3),
                require_nonzero_ordinary=True,
            ),
        ),
        expansion_checks=(
            ExpansionCheck(
                label="Gamma quadratic normal form in generators f1,f2",
                substitution={
                    x: t,
                    y: (U + t**3 + t**2 - 1) / t,
                    z: (V - ((U + t**3 + t**2 - 1) / t) ** 2) / t,
                },
                expected=-6 + A_205 * U**2 + B_205 * U * V + C_205 * V**2,
                normal_variables=(U, V),
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="Gamma curve",
                parameter=t,
                substitution={x: t, y: gamma_205_polynomial / t, z: -gamma_205_polynomial**2 / t**3},
                exclusions=(t, gamma_205_polynomial),
                value=-6,
            ),
        ),
    )

    cases["MM_2_6"] = Case(
        name="MM_2_6",
        potential=W_206,
        notes="One isolated Morse point and two disjoint positive-dimensional A1 components.",
        point_checks=(
            PointCheck(
                label="P1=(8/3,2,1)",
                point={x: sp.Rational(8, 3), y: 2, z: 1},
                value=28,
                ordinary_hessian_det=108,
            ),
        ),
        expansion_checks=(
            ExpansionCheck(
                label="Gamma_1 quadratic normal form",
                substitution={x: t, y: 1 + U, z: -1 + V},
                expected=-4 + t * U**2 - 2 * t * U * V + (t + 1) * V**2,
                normal_variables=(U, V),
            ),
            ExpansionCheck(
                label="Gamma_2 quadratic normal form",
                substitution={x: V + 2 * (t + 1) ** 2 / t, y: U - 2 / t, z: t},
                expected=-4 - t * (((t + 1) ** 2 * U + V) * (3 * (t + 1) ** 2 * U - V)) / (8 * (t + 1) ** 2),
                normal_variables=(U, V),
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="Gamma_1",
                parameter=t,
                substitution={x: t, y: 1, z: -1},
                exclusions=(t,),
                value=-4,
            ),
            ParametricBranchCheck(
                label="Gamma_2",
                parameter=t,
                substitution={x: 2 * (t + 1) ** 2 / t, y: -2 / t, z: t},
                exclusions=(t, t + 1),
                value=-4,
            ),
        ),
    )

    cases["MM_2_10"] = Case(
        name="MM_2_10",
        potential=W_210,
        notes="Two rational Morse points; saturated torus critical ideal is zero-dimensional.",
        identities=(("compact square W=F^2/y+x-4", W_210 - (F_210**2 / y + x - 4)),),
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(x - 64 * z + 16, 15 * y + 256 * z - 128, (16 * z - 5) * (16 * z - 3)),
            ),
        ),
        point_checks=(
            PointCheck(
                label="P1=(4,16/5,5/16)",
                point={x: 4, y: sp.Rational(16, 5), z: sp.Rational(5, 16)},
                value=20,
                ordinary_hessian_det=50,
            ),
            PointCheck(
                label="P2=(-4,16/3,3/16)",
                point={x: -4, y: sp.Rational(16, 3), z: sp.Rational(3, 16)},
                value=4,
                ordinary_hessian_det=-18,
            ),
        ),
    )

    cases["MM_2_12"] = Case(
        name="MM_2_12",
        potential=W_212,
        notes="Exact square completion with one A1 curve and one isolated Morse point.",
        identities=(("compact square W=F^2/y+xB^2-2", W_212 - (F_212**2 / y + x * B_212**2 - 2)),),
        point_checks=(
            PointCheck(
                label="P1=(3,2/3,1)",
                point={x: 3, y: sp.Rational(2, 3), z: 1},
                value=14,
                ordinary_hessian_det=64,
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="Gamma=(1,t,-1)",
                parameter=t,
                substitution={x: 1, y: t, z: -1},
                exclusions=(t,),
                value=-2,
            ),
        ),
    )

    cases["MM_2_13"] = Case(
        name="MM_2_13",
        potential=W_213,
        notes="A1 curve plus cubic algebraic isolated branch.",
        identities=(("compact square W=F^2/z+xB+y-2", W_213 - (F_213**2 / z + x * B_213 + y - 2)),),
        branch_checks=(
            ParametricBranchCheck(
                label="Gamma=(1,-1,t)",
                parameter=t,
                substitution={x: 1, y: -1, z: t},
                exclusions=(t,),
                value=-3,
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="xi^3-16 isolated branch",
                parameter=xi,
                minimal_polynomial=xi**3 - 16,
                point={x: xi, y: xi**2 / 4, z: 1},
                value=sp.Rational(3, 4) * xi**2 + 3 * xi + 2,
                ordinary_hessian_det=3 * (6 * xi**2 + 17 * xi + 32) / (8 * xi),
                coprime_tests=(xi, 6 * xi**2 + 17 * xi + 32),
            ),
        ),
    )

    cases["MM_2_16"] = Case(
        name="MM_2_16",
        potential=W_216,
        notes="Cubic algebraic isolated branch with nontrivial saturated critical ideal.",
        identities=(("compact square W=F^2/z+x+y-2", W_216 - (F_216**2 / z + x + y - 2)),),
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(
                    2 * x - 48 * z**2 + 125 * z - 76,
                    2 * y + 16 * z**2 - 31 * z + 12,
                    16 * z**3 - 63 * z**2 + 80 * z - 32,
                ),
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="beta^3-4beta-16 branch",
                parameter=beta,
                minimal_polynomial=q_beta,
                point={x: beta, y: beta**2 / 4 - 1, z: (92 - 4 * beta - 3 * beta**2) / 64},
                value=3 * beta + 4 / beta + 2,
                ordinary_hessian_det=(43 * beta**2 + 128 * beta + 208) / (4 * beta * (beta + 2)),
                coprime_tests=(beta, beta**2 - 4, 92 - 4 * beta - 3 * beta**2, 43 * beta**2 + 128 * beta + 208),
            ),
        ),
    )

    cases["MM_2_18"] = Case(
        name="MM_2_18",
        potential=W_218,
        notes="Quartic algebraic isolated branch with exact square completion and saturation.",
        identities=(("compact square W=x+yA^2-2", W_218 - (x + y * A_218**2 - 2)),),
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(x - 4 * z**2 - 4 * z, y - 1, 4 * z**4 + 4 * z**3 - 1),
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="eta quartic branch",
                parameter=eta,
                minimal_polynomial=q_eta,
                point={x: 4 * eta**2 + 4 * eta, y: 1, z: eta},
                value=2 * (4 * eta**2 + 6 * eta + 1),
                ordinary_hessian_det=32 * eta**4 * (eta + 1) ** 2 * (4 * eta + 3),
                coprime_tests=(eta, eta + 1, 4 * eta + 3),
            ),
        ),
    )

    cases["MM_2_19"] = Case(
        name="MM_2_19",
        potential=W_219,
        notes="Curve component plus quartic isolated branch; checks saturated decomposition data.",
        identities=(("two-square compact form", W_219 - ((x + 1) ** 2 / (y * z) + z * (x + y) ** 2 / x**2 + x + y * (x + 1) / x)),),
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(
                    x + y * z + y - z,
                    (y - 1) * (y + 2 * z**3 - z**2 - 2 * z - 3),
                    (y - 1) * (z**2 - z - 1) * (z**2 + z + 1),
                ),
            ),
        ),
        expansion_checks=(
            ExpansionCheck(
                label="Gamma normal quadratic form",
                substitution={x: -1 + U, y: 1 + V, z: t},
                expected=-1 + (t - 1 + 1 / t) * U**2 + (2 * t - 1) * U * V + t * V**2,
                normal_variables=(U, V),
            ),
        ),
        branch_checks=(
            ParametricBranchCheck(
                label="Gamma=(-1,1,t)",
                parameter=t,
                substitution={x: -1, y: 1, z: t},
                exclusions=(t,),
                value=-1,
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="rho quartic isolated branch",
                parameter=rho,
                minimal_polynomial=h_rho_219,
                point={x: rho**3 - rho**2 - 1, y: -2 * rho**3 + rho**2 + 2 * rho + 3, z: rho},
                ordinary_hessian_det=2 * (rho + 1) * (19 * rho**2 + 28 * rho + 10),
                coprime_tests=(rho, rho**3 - rho**2 - 1, -2 * rho**3 + rho**2 + 2 * rho + 3, (rho + 1) * (19 * rho**2 + 28 * rho + 10)),
            ),
        ),
    )

    cases["MM_2_29"] = Case(
        name="MM_2_29",
        potential=W_229,
        notes="One rational point plus a quartic algebraic branch.",
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(
                    5 * x + 64 * z**4 + 80 * z**3 - 32 * z**2 - 44 * z - 1,
                    5 * y - 96 * z**4 - 80 * z**3 + 48 * z**2 + 26 * z - 1,
                    (z + 1) * (16 * z**4 - 8 * z**2 - z + 1),
                ),
            ),
        ),
        point_checks=(
            PointCheck(
                label="rational point (1,-1,-1)",
                point={x: 1, y: -1, z: -1},
                value=-1,
                log_hessian_det=2,
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="alpha quartic branch",
                parameter=alpha,
                minimal_polynomial=q_alpha,
                point={x: alpha**2, y: alpha, z: alpha ** (-2)},
                value=alpha * (2 * alpha + 3),
                log_hessian_det=2 * (4 * alpha + 3),
                coprime_tests=(alpha, 4 * alpha + 3),
            ),
        ),
    )

    cases["MM_2_30"] = Case(
        name="MM_2_30",
        potential=W_230,
        notes="Quintic algebraic branch; checks the saturated ideal and log Hessian determinant modulo p(lambda).",
        saturation_checks=(
            SaturationCheck(
                label="critical ideal saturated by xyz",
                expected_generators=(
                    x - z**4 - 2 * z**3 - 3 * z**2 + z - 1,
                    y - z**4 - 2 * z**3 - 3 * z**2 + z - 1,
                    z**5 + z**4 + z**3 - 3 * z**2 + 3 * z - 1,
                ),
            ),
        ),
        parametric_point_checks=(
            ParametricPointCheck(
                label="lambda quintic branch",
                parameter=lam,
                minimal_polynomial=p_lam,
                point={x: lam * (lam + 1) / (1 - lam), y: lam * (lam + 1) / (1 - lam), z: lam},
                value=3 * lam**4 + 6 * lam**3 + 9 * lam**2 - 2 * lam + 3,
                log_hessian_det=2 * (2 * lam**4 + 2 * lam**3 + 2 * lam**2 - 7 * lam + 8),
                coprime_tests=(lam * (lam - 1) * (lam + 1), 2 * lam**4 + 2 * lam**3 + 2 * lam**2 - 7 * lam + 8),
            ),
        ),
    )

    branch_checks_81: List[ParametricBranchCheck] = []
    point_checks_81: List[PointCheck] = []
    normal_forms_81: List[NormalFormCheck] = []
    for eps in (1, -1):
        point_checks_81.append(
            PointCheck(
                label=f"P_{eps}=(eps,1,1)",
                point={x: eps, y: 1, z: 1},
                value=21 + 2 * eps,
                log_hessian_det=486 * eps,
            )
        )
        branch_checks_81.append(
            ParametricBranchCheck(
                label=f"Gamma_{eps}",
                parameter=t,
                substitution={x: eps, y: -1 / (t * (t + 1)), z: t},
                exclusions=(t, t + 1),
                value=2 * eps - 6,
                log_hessian_matrix=sp.Matrix([[2 * eps, 0, 0], [0, 0, 0], [0, 0, 0]]),
            )
        )
        normal_forms_81.append(
            NormalFormCheck(
                label=f"Gamma_{eps}: split quadratic plus residual A2",
                critical_value=2 * eps - 6,
                residual=u_81,
                exponent=3,
                unit=(z + 1) ** 2 / ((u_81 - 1) ** 2 * z),
                extra_term=x + 1 / x - 2 * eps,
            )
        )

    cases["MM_8_1"] = Case(
        name="MM_8_1",
        potential=W_81,
        notes="Two isolated Morse points and two residual A2 curve components.",
        point_checks=tuple(point_checks_81),
        branch_checks=tuple(branch_checks_81),
        normal_form_checks=tuple(normal_forms_81),
    )
    baseline_potentials = {
        "MM_2_7": x * (y + 1) ** 2 * ((1 + 1 / (x * y)) ** 2 + 1 / z) + z * (1 + 1 / (x * y)) ** 3 + (1 + 1 / (x * y)) - 5,
        "MM_2_9": x * (1 + 1 / z) + (y * z) * (1 + 1 / (y * z)) ** 3 / x + z * (1 + 1 / z) ** 2 * (y * z) * (1 + 1 / (y * z)) ** 2 - 1 / (y * z) - 4,
        "MM_2_11": x+((x)/(y*z))+y*z+y+z+((2)/(z))+((2)/(y))+((y*z)/(x))+((2*y)/(x))+((2*z)/(x))+((2)/(x))+((y)/(x*z))+((z)/(x*y)),
        "MM_2_14": x+((2*x)/(y))+((x)/(y**2))+y+z+((z)/(y))+((3)/(y))+((y**2)/(x*z))+((y)/(x))+((z)/(x))+((3)/(x))+((y)/(x**2)),
        "MM_2_15": x+((x)/(z))+((x)/(y*z))+y+((y)/(z))+z+((2)/(z))+((2)/(y))+((y)/(x*z))+((2)/(x))+((z)/(x*y)),
        "MM_2_17": ((x**2)/(y*z))+((x**2)/(y*z**2))+x+((2*x)/(z))+((x)/(y*z))+y+z+((2*z)/(x))+((1)/(x))+((z)/(x**2)),
        "MM_2_20": x+((2)/(x))+((1)/(y))+((1)/(z))+((x)/(y*z))+y+((y)/(x))+z+((2*z)/(x))+((z)/(x**2)),
        "MM_2_21": x+y+y**2*z+2*y*z+z+((x)/(y*z))+((2)/(y*z))+((1)/(x*y*z)),
        "MM_2_22": x+((x)/(y))+y+z+((1)/(y))+((1)/(y*z))+((z)/(x))+((2)/(x))+((1)/(x*z)),
        "MM_2_23": x**2*y+2*x*y+x+y+z+((2)/(x*y))+((1)/(x**2*y**2*z)),
        "MM_2_24": y+((2*y)/(x*z))+((y)/(x**2*z**2))+z+((z)/(y))+x+((1)/(x))+((2)/(x*z))+((1)/(y)),
        "MM_2_25": x+((x)/(y))+y+z+((1)/(y*z))+((y*z)/(x))+((2)/(x))+((y*z)/(x**2)),
        "MM_2_26": x+((z)/(x))+((2)/(x))+((1)/(x*z))+y+((y)/(z))+z+((1)/(y)),
        "MM_2_27": x+((x)/(z))+y+z+((1)/(y*z))+((1)/(x))+((1)/(x*y)),
        "MM_2_28": x*y*z**2+x*y*z+x+y+z+((1)/(y*z))+((1)/(x*z)),
        "MM_2_31": x+((x)/(y))+y+z+((1)/(y*z))+((1)/(x)),
        "MM_2_32": ((x*y)/(z))+((x)/(z))+((y)/(z))+((2)/(z))+((1)/(x*z))+((1)/(y*z))+((1)/(x*y*z))+z,
        "MM_2_33": x+((x)/(z))+y+z+((1)/(x*y)),
        "MM_2_34": x+((1)/(x))+y+z+((1)/(y*z)),
        "MM_2_35": x+((x)/(y*z))+y+z+((1)/(x)),
        "MM_2_36": ((x**2)/(y*z))+x+y+z+((1)/(x)),
        "MM_3_1": ((x**2*y**3)/(z))+((4*x*y**2)/(z))+((6*y)/(z))+((4)/(x*z))+((1)/(x**2*y*z))+x*y**2+3*x*y+x+2*y+((2)/(y))+((1)/(x))+((3)/(x*y))+((1)/(x*y**2))+z+((2*z)/(y))+((z)/(y**2)),
        "MM_3_2": x*y*z**2+x*y*z+3*x*z+x+((3*x)/(y))+((x)/(y**2*z))+3*y*z+y+z+((1)/(y))+((3)/(y*z))+((3*y)/(x))+((1)/(x))+((3)/(x*z))+((y)/(x**2*z)),
        "MM_3_3": ((x*y)/(z))+x+((x)/(z))+((2*x)/(y))+((x)/(y**2))+y+((y)/(z))+z+((3)/(y))+((2*y)/(x))+((3)/(x))+((y)/(x**2)),
        "MM_3_4": x+((x)/(y))+2*y*z+y+z+((2)/(y))+((2)/(y*z))+((1)/(y**2*z))+((y**2*z**2)/(x))+((4*y*z)/(x))+((6)/(x))+((4)/(x*y*z))+((1)/(x*y**2*z**2)),
        "MM_3_5": x*y*z+x*z**2+2*x*z+x+y+2*z+((2)/(z))+((1)/(y))+((1)/(y*z))+((1)/(x))+((2)/(x*z))+((1)/(x*z**2)),
        "MM_3_6": x+((x*z)/(y))+((2*x)/(y))+((x)/(y*z))+y+z+((1)/(y))+((1)/(y*z))+((2*y)/(x))+((2)/(x))+((y)/(x**2)),
        "MM_3_7": x*y/z+x+x/z+x/y+y+z+2/y+y/x+2/x+1/(x*y),
        "MM_3_8": x+((x)/(z))+y+z+((1)/(z))+((2)/(y))+((1)/(y*z))+((y)/(x))+((3)/(x))+((3)/(x*y))+((1)/(x*y**2)),
        "MM_3_9": ((x**2)/(y*z))+x+((4*x)/(y*z))+y+z+((6)/(y*z))+((1)/(x))+((4)/(x*y*z))+((1)/(x**2*y*z)),
        "MM_3_10": ((x*y)/(z))+x+((x)/(y))+y+z+((2)/(y))+((y)/(x))+((2)/(x))+((1)/(x*y)),
        "MM_3_11": x+((x)/(z))+((x)/(y*z))+y+z+((1)/(y))+((y)/(x))+((2)/(x))+((1)/(x*y)),
        "MM_3_12": x*y+x+y+z+((1)/(y))+((y)/(x*z))+((3)/(x))+((2)/(x*z))+((3)/(x**2*z))+((1)/(x**3*z**2)),
        "MM_3_13": x+y+z+((2)/(y*z))+((y*z)/(x))+((3)/(x))+((3)/(x*y*z))+((1)/(x*y**2*z**2)),
        "MM_3_14": ((x**2)/(y*z))+x+((3*x)/(y*z))+y+z+((3)/(y*z))+((1)/(x))+((1)/(x*y*z)),
        "MM_3_15": x+((x)/(y*z))+y+z+((1)/(y))+((y)/(x))+((2)/(x))+((1)/(x*y)),
        "MM_3_16": x+((x)/(y))+y+((y)/(z))+z+((1)/(y))+((y)/(x*z))+((1)/(x)),
        "MM_3_17": x+y+z+((1)/(y))+((2)/(y*z))+((1)/(x))+((2)/(x*y*z))+((1)/(x*y**2*z**2)),
        "MM_3_18": x+((x)/(y))+y+z+((z)/(x))+((2)/(x))+((1)/(x*z)),
        "MM_3_19": x+((2*x)/(y*z))+((x)/(y**2*z**2))+y+z+((2)/(y*z))+((1)/(x)),
        "MM_3_20": x+y+z+((1)/(y*z))+((y)/(x))+((2)/(x))+((1)/(x*y)),
        "MM_3_21": x+y+z+((1)/(y))+((y)/(x))+((2)/(x))+((1)/(x*y*z))+((1)/(x**2*z)),
        "MM_3_22": ((x**2)/(y*z))+x+((2*x)/(y*z))+y+z+((1)/(y*z))+((1)/(x)),
        "MM_3_23": x*z+x+((x)/(y))+y+z+((1)/(y*z))+((1)/(x)),
        "MM_3_24": x+y+z+((1)/(y))+((y)/(x))+((1)/(x))+((1)/(x*y*z)),
        "MM_3_25": x+((x)/(z))+y+z+((1)/(x))+((1)/(x*y)),
        "MM_3_26": x*y+x+y+z+((1)/(y*z))+((1)/(x)),
        "MM_3_27": x+y+z+((1)/(x))+((1)/(y))+((1)/(z)),
        "MM_3_28": x+((x)/(z))+y+z+((1)/(y))+((1)/(x)),
        "MM_3_29": x*y+x+((x)/(y*z))+y+z+((1)/(x)),
        "MM_3_30": x+((x)/(y))+y+((y)/(z))+z+((1)/(x)),
        "MM_3_31": ((x**2)/(y**2*z))+x+((2*x)/(y))+y+z+((1)/(x)),
        "MM_4_1": x+y*z**2+2*y*z+y+2*z+((1)/(y))+((2)/(y*z))+((1)/(x))+((2)/(x*y*z))+((1)/(x*y**2*z**2)),
        "MM_4_2": x+y+((y)/(z))+z+((1)/(z))+((2)/(y))+((y)/(x))+((3)/(x))+((3)/(x*y))+((1)/(x*y**2)),
        "MM_4_3": x+((2*x)/(y*z))+((x)/(y**2*z**2))+y*z**2+2*y*z+y+((2)/(y*z))+((1)/(x)),
        "MM_4_4": x+((x)/(y))+y+z+((1)/(z))+((1)/(y))+((2)/(x))+((2)/(x*z))+((1)/(x**2*z)),
        "MM_4_5": x+((x)/(z))+y+z+((1)/(y))+((2*y)/(x))+((2)/(x))+((y)/(x**2)),
        "MM_4_6": x+((x)/(y))+y+z+((1)/(y))+((y)/(x))+((2)/(x))+((y)/(x**2*z))+((1)/(x**2*z)),
        "MM_4_7": x+((x)/(y))+y+z+((1)/(y))+((z)/(x))+((2)/(x))+((1)/(x*z)),
        "MM_4_8": x+y+z+((1)/(z))+((z)/(y))+((1)/(y))+((1)/(x))+((1)/(x*z)),
        "MM_4_9": x*y+x+y+z+((1)/(y))+((2)/(x))+((1)/(x**2*z)),
        "MM_4_10": x*y+x+y+z+((1)/(y))+((1)/(y*z))+((1)/(x)),
        "MM_4_11": x+y+z+((1)/(y))+((z)/(x))+((2)/(x))+((1)/(x*z)),
        "MM_4_12": x*y+x+((x)/(z))+y+z+((1)/(y))+((1)/(x)),
        "MM_4_13": x*y + x*y/z + x + y + z + 1/y + 1/x,
        "MM_5_1": x+y+((y)/(z))+z+((2)/(y))+((y)/(x))+((3)/(x))+((3)/(x*y))+((1)/(x*y**2)),
        "MM_5_2": x+y+((y)/(z))+z+((1)/(y))+((y)/(x))+((2)/(x))+((1)/(x*y)),
        "MM_5_3": x+y+z+((1)/(y))+((3)/(x))+((2)/(x*z))+((3)/(x**2*z))+((1)/(x**3*z**2)),
        "MM_6_1": x+y+z+((1)/(z))+((2)/(y))+((y)/(x))+((3)/(x))+((3)/(x*y))+((1)/(x*y**2)),
        "MM_7_1": x*z**2+2*x*z+x+y+4*z+((1)/(y))+((6)/(x))+((2)/(x*z))+((4)/(x**2*z))+((1)/(x**3*z**2)),
    }
    baseline_saturation_generators = {
        "MM_2_9": (
            19*x + 49*z**3 - 115*z**2 - 36*z + 88,
            2*y + 5*z**3 - 12*z**2 - 4*z + 12,
            z**4 - 2*z**3 - 2*z**2 + 2*z + 2,
        ),
        "MM_2_11": (
            x - z**2 - z,
            y - z,
            z**4 + 2*z**3 + z**2 - 2*z - 3,
        ),
        "MM_2_14": (
            638*x + 1507*z**3 - 6420*z**2 + 3039*z - 378,
            29*y + 33*z**3 - 101*z**2 - 5*z + 4,
            11*z**4 - 52*z**3 + 48*z**2 - 12*z + 1,
        ),
        "MM_2_15": (
            x + y - z**4 + 2*z**3 + z**2 - 2*z - 1,
            (z + 1)*(y + z**3 - z**2 - 3*z - 1),
            (z + 1)*(z**4 - 3*z**2 - 4*z - 1),
        ),
        "MM_2_17": (
            7*x - 88*z**3 + 95*z**2 - 75*z + 50,
            7*y + 286*z**3 - 350*z**2 + 279*z - 192,
            11*z**4 - 16*z**3 + 14*z**2 - 10*z + 2,
        ),
        "MM_2_20": (
            17558*x + 1761*z**5 - 23474*z**4 + 20398*z**3 - 45568*z**2 + 12618*z + 2764,
            17558*y + 1537*z**5 - 21196*z**4 + 26368*z**3 - 34278*z**2 - 2836*z + 7956,
            z**6 - 14*z**5 + 20*z**4 - 26*z**3 + 8*z**2 + 10*z + 2,
        ),
        "MM_2_21": (
            90*x - 176*z**3 + 13*z**2 + 307*z - 26,
            30*y + 32*z**3 + 34*z**2 - 79*z - 68,
            (z**2 - 2)*(16*z**2 + 9*z + 4),
        ),
        "MM_2_22": (
            x + 2*z**4 - 2*z**3 + 2*z**2 - z - 2,
            y - 4*z**4 + 2*z**3 - 6*z**2 + 3,
            2*z**5 + 2*z**3 + z**2 - 2*z - 1,
        ),
        "MM_2_23": (
            7*x - 32*z**2 + 4*z + 24,
            119*y - 160*z**2 - 92*z + 92,
            16*z**3 + 16*z**2 - 16*z - 17,
        ),
        "MM_2_24": (
            x**2 - 2*x*z - 1,
            y - 1,
            (z - 1)*(z**2 + z + 1),
        ),
        "MM_2_25": (
            5*x - 16*z**2 - 11*z + 2,
            5*y + 4*z**2 - 11*z - 3,
            (2*z - 1)*(2*z**2 + 2*z + 1),
        ),
        "MM_2_26": (
            2*x - 2*z**5 - z**4 - 5*z**3 - 7*z**2 + 2*z + 5,
            2*y - z**5 + z**4 - 3*z**3 + 3*z - 2,
            z**6 + 2*z**4 + 2*z**3 - 3*z**2 - 2*z + 1,
        ),
        "MM_2_27": (
            x - z**5 + z**2 + z,
            y - z,
            (z**3 - z - 1)*(z**3 + z - 1),
        ),
        "MM_2_28": (
            2*x - y**2*z**4 + y**2*z**3 - y**2*z**2 + y**2*z + y*z**5 - y*z + 2*y + z**3 - z**2 - z - 3,
            (y + 2*z**3 - z**2 - 4)*(y**2*z - y*z**2 - y*z - 1),
            (z**4 - 2*z - 1)*(y**2*z - y*z**2 - y*z - 1),
        ),
        "MM_2_31": (
            x + z**5 - 3*z**4 + 2*z**3 - z**2 + z - 1,
            y + z**4 - 2*z**3 - 2*z + 1,
            z**6 - 2*z**5 - 2*z**3 + z**2 + 1,
        ),
        "MM_2_32": (
            x - y,
            3*y**2 + 3*y - z**2 + 2,
            (y - 1)*(z**2 + 1),
            (z**2 - 8)*(z**2 + 1),
        ),
        "MM_2_33": (
            x - z**2,
            y - z**2 - z,
            (z**3 + z**2 - 1)*(z**3 + z**2 + 1),
        ),
        "MM_2_34": (
            (x - 1)*(x + 1),
            y - z,
            (z - 1)*(z**2 + z + 1),
        ),
        "MM_2_35": (
            x - z**3,
            y - z,
            z**6 + z**4 - 1,
        ),
        "MM_2_36": (
            2*x + z**5 - 4*z**4 - z**2,
            y - z,
            z**6 - 4*z**5 - 2*z**3 + 1,
        ),
        "MM_3_1": (
            3*x**2 + 3*x*y*z - 6*x*y - 6*x - 2*y**3*z + y**3 - y**2*z + 2*y**2 - y*z + 5*y + z + 1,
            (x - 1)*(y**2 + y + 1),
            (y - 1)*(y + 1)*(y**2 + y + 1),
            (y + 1)*(z - 2)*(z + 2)*(y**2 + y + 1),
        ),
        "MM_3_3": (
            247*x + 247*y + 207*z**3 - 539*z**2 - 228*z + 240,
            247*y**2 + 3952*y + 981*z**3 - 3188*z**2 - 1596*z + 2512,
            (z - 4)*(26*y - 9*z**2 - 12*z + 32),
            (z - 4)*(9*z**3 - 16*z**2 - 16),
        ),
        "MM_3_4": (
            x - 32*z**5 - 16*z**4 + 32*z**3 + 24*z**2 - 2*z - 7,
            y + 16*z**4 - 8*z**2 - 8*z + 1,
            (z - 1)*(16*z**5 - 8*z**3 - 8*z**2 + z + 1),
        ),
        "MM_3_5": (
            x - 4*z**7 - 15*z**6 - 12*z**5 + 20*z**4 + 36*z**3 + 6*z**2 - 16*z - 13,
            y - z**7 - 4*z**6 - 4*z**5 + 4*z**4 + 9*z**3 + 2*z**2 - 3*z - 2,
            (z**4 + 2*z**3 - 3*z - 1)*(z**4 + 2*z**3 - z - 1),
        ),
        "MM_3_6": (
            95*x + 128*z**5 + 343*z**4 + 432*z**3 + 1186*z**2 - 67*z - 1570,
            285*y - 131*z**5 - 350*z**4 - 473*z**3 - 1318*z**2 - 129*z + 1236,
            (z**3 + 3*z**2 - 3)*(z**3 + z**2 + 4*z + 5),
        ),
        "MM_3_7": (2*x + 3*z - 7, y + 3*z - 6, z**2 - 6*z + 7),
        "MM_3_8": (
            4*x - z**5 - z**4 + 4*z**3 + z + 1,
            4*y - z**7 - 2*z**6 + 2*z**5 + 5*z**4 + 7*z**3 + 4*z**2 + 5,
            (z + 1)*(z**7 + z**6 - 3*z**5 - 3*z**4 - 5*z**3 + 3*z**2 - z - 1),
        ),
        "MM_3_9": (
            4*x**2 + 40*x + z**3 - 60,
            (x - 1)*(z - 4),
            y - z,
            (z - 4)*(z**3 - 16),
        ),
        "MM_3_10": (
            49*x + 49*y + 36*z**4 - 130*z**3 - 36*z**2 + 14*z + 65,
            49*y**2 + 441*y + 160*z**4 - 584*z**3 - 209*z**2 + 63*z + 292,
            (z - 4)*(7*y - 8*z**3 + z + 4),
            (z - 4)*(2*z - 1)*(2*z**3 - 2*z - 1),
        ),
        "MM_3_11": (
            27699*x - 131*z**5 + 2152*z**4 - 7530*z**3 - 6754*z**2 - 15798*z + 21285,
            27699*y + 2155*z**5 - 21446*z**4 + 18150*z**3 - 42613*z**2 + 68316*z + 4866,
            z**6 - 10*z**5 + 10*z**4 - 32*z**3 + 49*z**2 - 12*z + 43,
        ),
        "MM_3_12": (
            1593499155*x - 702272339*z**7 + 40707585955*z**6 + 20317860746*z**5 - 238219059006*z**4 + 244020218545*z**3 - 92921979643*z**2 + 6975254335*z + 5521924046,
            1593499155*y - 113358064*z**7 + 6655734902*z**6 - 1595553041*z**5 - 43476512952*z**4 + 65780613764*z**3 - 30172214777*z**2 + 2541159668*z + 890703988,
            (z**4 - 59*z**3 + 37*z**2 + 9*z + 1)*(z**4 + z**3 - 5*z**2 + 3*z - 1),
        ),
        "MM_3_13": (
            48*x - 125,
            3*y - 4,
            3*z - 4,
        ),
        "MM_3_14": (
            12*x + z**4 - 2*z**3 - 6*z + 3,
            y - z,
            z**6 - 4*z**5 + z**4 - 12*z**3 + 27*z**2 + 27,
        ),
        "MM_3_15": (
            1759*x + 128*z**6 - 548*z**5 - 18*z**4 - 2192*z**3 - 873*z**2 - 310*z + 1812,
            3518*y + 1502*z**6 - 7200*z**5 - 2355*z**4 + 2862*z**3 - 4280*z**2 + 2244*z + 4882,
            2*z**7 - 12*z**6 + 11*z**5 - 8*z**4 + 4*z**3 + 4*z**2 + 4*z - 4,
        ),
        "MM_3_16": (
            61*x + 54*z**7 + 6*z**6 + 95*z**5 - 165*z**4 - 93*z**3 - 138*z**2 + 5*z + 62,
            61*y - 38*z**7 + 50*z**6 - 103*z**5 + 211*z**4 - 165*z**3 + 131*z**2 - 121*z + 49,
            2*z**8 + 5*z**6 - 6*z**5 + z**4 - 10*z**3 - 4*z + 1,
        ),
        "MM_3_17": (
            4*x - z - 4,
            4*y - z**2 - 4*z,
            z**4 + 4*z**3 - 16,
        ),
        "MM_3_18": (
            x + 3*z**6 - 2*z**5 + 11*z**4 - 8*z**3 + 5*z**2 + 6*z - 7,
            y - 4*z**6 + 2*z**5 - 15*z**4 + 8*z**3 - 8*z**2 - 9*z + 8,
            z**7 - z**6 + 4*z**5 - 4*z**4 + 3*z**3 + z**2 - 3*z + 1,
        ),
        "MM_3_19": (
            17*x - z**2 + 4*z - 16,
            y - z,
            z**3 - 4,
        ),
        "MM_3_20": (
            x**2 + z**2 - 2*z - 3,
            (z - 1)*(x + z + 2),
            y + 2*z**2 - 2*z - 1,
            (z - 1)*(2*z**2 + 2*z + 1),
        ),
        "MM_3_21": (
            1627*x + 4400*z**7 + 18777*z**6 + 17991*z**5 + 74273*z**4 + 45483*z**3 - 79913*z**2 - 41283*z + 18623,
            1627*y - 6526*z**7 - 27988*z**6 - 27439*z**5 - 111760*z**4 - 71655*z**3 + 113203*z**2 + 55072*z - 24229,
            z**8 + 4*z**7 + 3*z**6 + 16*z**5 + 6*z**4 - 20*z**3 - 4*z**2 + 6*z - 1,
        ),
        "MM_3_22": (
            110*x + 4*z**6 - 13*z**5 + 4*z**4 - 68*z**3 + 36*z**2 + 4*z + 58,
            y - z,
            z**7 - 4*z**6 - 4*z**4 + 8*z**3 + 8*z**2 - 4,
        ),
        "MM_3_23": (
            205*x + 293*z**7 - 795*z**6 - 56*z**5 - 2211*z**4 - 1878*z**3 + 953*z**2 + 435*z + 91,
            205*y - 309*z**7 + 865*z**6 - 27*z**5 + 2408*z**4 + 1679*z**3 - 1089*z**2 - 775*z - 238,
            z**8 - 2*z**7 - 2*z**6 - 8*z**5 - 12*z**4 - 2*z**3 + 3*z**2 + 2*z + 1,
        ),
        "MM_3_24": (
            4*x + 7*z**7 - 58*z**6 + 47*z**5 - 62*z**4 - 5*z**3 + 12*z**2 - 13*z + 12,
            4*y - 7*z**7 + 59*z**6 - 55*z**5 + 67*z**4 - 7*z**3 - 13*z**2 + 5*z - 13,
            z**8 - 8*z**7 + 4*z**6 - 4*z**5 - 6*z**4 + 4*z**3 + 1,
        ),
        "MM_3_25": (
            x - z**2,
            y - z**6 - 2*z**5 - z**4 + z**2 + z + 1,
            (z - 1)*(z + 1)*(z**2 + 1)*(z**2 + z - 1)*(z**2 + z + 1),
        ),
        "MM_3_26": (
            x - z**3 + 1,
            y + z**6 + z**4 - 2*z**3 - 2*z,
            z**8 + z**6 - 2*z**5 - 2*z**3 + 1,
        ),
        "MM_3_27": (
            (x - 1)*(x + 1),
            (y - 1)*(y + 1),
            (z - 1)*(z + 1),
        ),
        "MM_3_28": (
            x - z**2,
            (y - 1)*(y + 1),
            z**4 + z**3 - 1,
        ),
        "MM_3_29": (
            23*x + 6*z**7 - 29*z**6 + 28*z**5 - 43*z**4 + 3*z**3 + 10*z**2 - 16*z + 21,
            23*y + 16*z**7 - 85*z**6 + 90*z**5 - 61*z**4 + 54*z**3 + 42*z**2 - 81*z + 10,
            z**8 - 4*z**7 - 2*z**5 - z**4 + 4*z**3 + 1,
        ),
        "MM_3_30": (
            x - z**4 - z**3,
            y - z**2,
            z**8 + 2*z**7 + 2*z**6 + 2*z**5 + z**4 - 1,
        ),
        "MM_3_31": (
            x - 4*z**2,
            y - 4*z,
            16*z**4 + 16*z**3 - 1,
        ),
        "MM_4_1": (
            x*y + x - 2*y*z - y - 2,
            x*z - 1,
            (y - 2)*(y*z + y + 1),
            (2*z - 1)*(y*z + y + 1),
        ),
        "MM_4_2": (
            15*x - 26*z**5 - 8*z**4 + 54*z**3 - 7*z**2 + 39*z - 14,
            y - z**2 + 1,
            (z**3 - 3*z + 1)*(z**3 + 2*z**2 + z + 1),
        ),
        "MM_4_3": (
            2*x*y - 2*x + 2*y - z - 1,
            (2*x - 1)*(z + 1),
            (y - 1)*(z + 1),
            (z - 1)*(z + 1),
        ),
        "MM_4_4": (
            x + 16*z**4 - 48*z**3 + 40*z**2 - 8*z,
            y - 16*z**4 + 48*z**3 - 44*z**2 + 12*z + 1,
            16*z**5 - 64*z**4 + 88*z**3 - 48*z**2 + 8*z + 1,
        ),
        "MM_4_5": (
            x - z**2,
            20*y - 9*z**4 - 11*z**3 + 4*z**2 + 8*z + 12,
            (z + 1)*(z**4 + z**3 - 4),
        ),
        "MM_4_6": (
            456761*x + 499916*z**9 + 1624217*z**8 - 10282244*z**7 - 4445956*z**6 + 5337383*z**5 - 1974010*z**4 + 3373121*z**3 - 90990*z**2 - 1879996*z + 1125167,
            456761*y - 597994*z**9 - 1956497*z**8 + 12259824*z**7 + 5701293*z**6 - 6016682*z**5 + 591865*z**4 - 5688312*z**3 - 428082*z**2 + 1682856*z - 708578,
            z**10 + 4*z**9 - 18*z**8 - 24*z**7 + z**6 + 4*z**5 + 8*z**4 + 6*z**3 - 2*z**2 + 1,
        ),
        "MM_4_7": (
            x + 8*z**4 - 2*z**3 - 9*z**2 + 18*z - 10,
            y - 18*z**4 - z**3 + 17*z**2 - 36*z + 17,
            (2*z - 1)*(z**2 - z + 1)*(z**2 + z - 1),
        ),
        "MM_4_8": (
            2*x - 2*y + z**8 + z**7 - 2*z**6 - 7*z**5 - 6*z**4 - z**3 + 7*z**2 + 4*z + 3,
            y**2 - z - 1,
            (z - 1)*(2*y - z**7 - 2*z**6 + 2*z**5 + 9*z**4 + 9*z**3 + 2*z**2 - 3*z - 3),
            (z - 1)*(z**8 + 2*z**7 - z**6 - 8*z**5 - 11*z**4 - 8*z**3 - z**2 + 2*z + 1),
        ),
        "MM_4_9": (
            x + 16*z**4 + 16*z**3 - 8*z**2 - 8*z,
            y - 4*z**2 + 1,
            16*z**5 + 16*z**4 - 8*z**3 - 8*z**2 + 1,
        ),
        "MM_4_10": (
            x - z**4 - z**3 + 1,
            y + z**8 + 2*z**7 + 2*z**6 + 2*z**5 - z**4 - 2*z**3 - 2*z**2 - 2*z,
            z**10 + 2*z**9 + 2*z**8 + 2*z**7 - z**6 - 2*z**5 - 2*z**4 - 2*z**3 + 1,
        ),
        "MM_4_11": (
            x - 2*z**2 + z - 2,
            (y - 1)*(y + 1),
            z**3 - z**2 + 2*z - 1,
        ),
        "MM_4_12": (
            x - z**2,
            y + z**9 + z**8 - z**7 + 2*z**6 - z**5 + 2*z**3 - z**2 + z - 2,
            z**10 + 2*z**9 + z**8 + 2*z**7 - z**6 - 2*z**5 - 2*z**4 - 2*z**3 + z**2 + 1,
        ),
        "MM_4_13": (
            x - y*z**2 - z**4 - z**3,
            y**2 + y*z**2 + y*z - 1,
            (z + 1)*(z**2 + 1)*(y - z**3 + z - 1),
            (z + 1)*(z**2 + 1)*(z**2 - z + 1)*(z**2 + z - 1),
        ),
        "MM_5_1": (
            16*x + 23*z**4 + 2*z**3 - 24*z**2 + 16*z - 71,
            y - z**2,
            z**5 + 2*z**4 - z - 4,
        ),
        "MM_5_2": (
            x - 2*z**7 - 5*z**6 - 4*z**5 - 4*z**4 - 2*z**3 + z**2 + 3*z + 3,
            y - z**2,
            (z**4 + z - 1)*(z**4 + 2*z**3 - z - 1),
        ),
        "MM_5_3": (
            3*x - 8,
            (y - 1)*(y + 1),
            8*z - 9,
        ),
        "MM_6_1": (
            x - y - 1,
            y**2 - y - 1,
            (z - 1)*(z + 1),
        ),
        "MM_7_1": (
            (x - 4)*(x**2*z**2 + x**2*z + 2*x*z + 1),
            (4*z - 1)*(x**2*z**2 + x**2*z + 2*x*z + 1),
            (y - 1)*(y + 1),
        ),
    }

    detailed_note = "Detailed Mori--Mukai validation with fixed symbolic checks transcribed from main.tex."
    saturation_note = "Detailed Mori--Mukai validation: saturated torus critical ideal is checked against fixed generators."

    for name, potential in baseline_potentials.items():
        if name in cases:
            continue

        if name == "MM_2_7":
            D_207 = t**2 + t + 1
            cases[name] = Case(
                name=name,
                potential=potential,
                notes=detailed_note + " Checks the Gamma curve and the cubic isolated branch.",
                parametric_point_checks=(
                    ParametricPointCheck(
                        label="alpha cubic isolated branch",
                        parameter=alpha,
                        minimal_polynomial=4 * alpha**3 - 4 * alpha**2 - 4 * alpha + 3,
                        point={x: alpha, y: 1, z: -28 * alpha**2 + 56 * alpha - 24},
                        value=-(84 * alpha**2 - 239 * alpha + 114) / ((5 * alpha - 6) * (5 * alpha - 3)),
                        log_hessian_det=2 * (10156 * alpha**2 + 3307 * alpha - 5727) / (7700 * alpha**2 - 14855 * alpha + 6171),
                        coprime_tests=(
                            alpha,
                            -28 * alpha**2 + 56 * alpha - 24,
                            10156 * alpha**2 + 3307 * alpha - 5727,
                            7700 * alpha**2 - 14855 * alpha + 6171,
                        ),
                    ),
                ),
                branch_checks=(
                    ParametricBranchCheck(
                        label="Gamma curve",
                        parameter=t,
                        substitution={
                            x: -D_207 / (t * (t + 1) ** 2),
                            y: t,
                            z: -D_207**2 / t**2,
                        },
                        exclusions=(t, t + 1, D_207),
                        value=-5,
                    ),
                ),
            )
            continue

        if name == "MM_3_2":
            U_302 = 1 + (x + y) / (x * y * z)
            cases[name] = Case(
                name=name,
                potential=potential,
                notes=detailed_note + " Checks the Morse--Bott curve and the two cubic isolated branches.",
                identities=(
                    ("compact U form", potential - (x * y * z**2 * U_302**3 + z * (x * y + 1) * U_302 - 6)),
                ),
                parametric_point_checks=(
                    ParametricPointCheck(
                        label="positive cubic isolated branch",
                        parameter=lam,
                        minimal_polynomial=lam**3 + 4 * lam**2 - 4,
                        point={x: 1, y: 1, z: lam},
                        value=3 * lam**2 + 16 * lam + 10,
                        log_hessian_det=8 * (22 * lam**2 + 107 * lam + 96),
                        coprime_tests=(lam, 22 * lam**2 + 107 * lam + 96),
                    ),
                    ParametricPointCheck(
                        label="negative cubic isolated branch",
                        parameter=rho,
                        minimal_polynomial=rho**3 - 2 * rho**2 + 4,
                        point={x: -1, y: -1, z: rho},
                        value=3 * rho**2 - 8 * rho + 2,
                        log_hessian_det=-8 * (19 * rho**2 - 59 * rho + 66),
                        coprime_tests=(rho, 19 * rho**2 - 59 * rho + 66),
                    ),
                ),
                branch_checks=(
                    ParametricBranchCheck(
                        label="Gamma Morse--Bott curve",
                        parameter=t,
                        substitution={x: t, y: -1 / t, z: t - 1 / t},
                        exclusions=(t, t - 1, t + 1),
                        value=-6,
                        log_hessian_matrix=sp.Matrix(
                            [
                                [2 / t, -t + 1 / t, -t + 1 / t],
                                [-t + 1 / t, -2 * t, -t + 1 / t],
                                [-t + 1 / t, -t + 1 / t, 0],
                            ]
                        ),
                    ),
                ),
            )
            continue

        if name == "MM_3_7":
            cases[name] = Case(
                name=name,
                potential=potential,
                notes="CCGK Appendix A polynomial: two isolated Morse points.",
                saturation_checks=(
                    SaturationCheck(
                        label="critical ideal saturated by xyz",
                        expected_generators=baseline_saturation_generators[name],
                    ),
                ),
                parametric_point_checks=(
                    ParametricPointCheck(
                        label="z^2-6z+7 isolated branch",
                        parameter=t,
                        minimal_polynomial=t**2 - 6*t + 7,
                        point={x: (7 - 3*t)/2, y: 6 - 3*t, z: t},
                        value=18 - 4*t,
                        log_hessian_det=-6*(9*t - 35),
                        coprime_tests=(t*(t-2)*(7-3*t)*(2-t), 9*t-35),
                    ),
                ),
            )
            continue

        generators = baseline_saturation_generators[name]
        cases[name] = Case(
            name=name,
            potential=potential,
            notes=saturation_note,
            saturation_checks=(
                SaturationCheck(
                    label="critical ideal saturated by xyz",
                    expected_generators=generators,
                ),
            ),
        )

    return cases


CASES = build_cases()


def normalize_manuscript_label(raw: str) -> str:
    return "MM_" + raw.replace("-", "_")


def case_sort_key(label: str) -> Tuple[int, int]:
    _prefix, rank, number = label.split("_")
    return int(rank), int(number)


def manuscript_labels(path: Path = Path("main.tex")) -> List[str]:
    """Return Mori--Mukai labels found in main.tex, normalized as MM_2_10."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    labels = {normalize_manuscript_label(match) for match in re.findall(r"MM_\{([0-9]+-[0-9]+)\}", text)}
    return sorted(labels, key=case_sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected Mori--Mukai LG computations.")
    parser.add_argument("--list", action="store_true", help="List available cases and exit.")
    parser.add_argument("--coverage", action="store_true", help="Compare implemented cases with Mori--Mukai labels found in main.tex.")
    parser.add_argument("--case", action="append", choices=sorted(CASES, key=case_sort_key), help="Case to run; repeatable.")
    parser.add_argument("--all", action="store_true", help="Run all available cases.")
    parser.add_argument("--verbose", action="store_true", help="Print log systems and saturated generators.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for name in sorted(CASES, key=case_sort_key):
            print(name)
        return 0

    if args.coverage:
        found = manuscript_labels()
        implemented = sorted(CASES, key=case_sort_key)
        detailed = sorted((name for name, case in CASES.items() if case_has_assertive_checks(case)), key=case_sort_key)
        baseline = sorted((name for name, case in CASES.items() if not case_has_assertive_checks(case)), key=case_sort_key)
        missing = [label for label in found if label not in CASES]
        print("Implemented cases:")
        for name in implemented:
            marker = "detailed" if name in detailed else "baseline"
            print(f"  {name} ({marker})")
        print(f"\nFound in main.tex: {len(found)} Mori--Mukai labels")
        print(f"Implemented entries: {len(implemented)}")
        print(f"Detailed checked entries: {len(detailed)}")
        print(f"Baseline potential-only entries: {len(baseline)}")
        print(f"Missing data entries: {len(missing)}")
        if missing:
            print("\nMissing:")
            for label in missing:
                print(f"  {label}")
        return 0

    selected = sorted(CASES, key=case_sort_key) if args.all else args.case
    if not selected:
        print("No case selected. Use --list, --case NAME, or --all.")
        return 2

    for name in selected:
        check_case(CASES[name], verbose=args.verbose)

    print("\nAll selected Mori--Mukai checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())