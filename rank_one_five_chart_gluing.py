#!/usr/bin/env python3
"""Compactification audit for the rank-one comparison.

The four genuine projective charts are X, Y, Z, and T nonzero in P^3.  The
fifth computational chart is the dense torus inside the T-chart.  It is
redundant as an open cover, but records exactly which critical points are
already visible before compactification and prevents multiple counting when
the chartwise calculations are glued.  The final two checks treat the
compactified V_6 and V_22 pencils directly, without a source-torus census.
"""

from __future__ import annotations

import sympy as sp


X, Y, Z, T, lam = sp.symbols("X Y Z T lambda")
x, y, z, t, a = sp.symbols("x y z t a")
u, v, w, q = sp.symbols("u v w q")
PROJECTIVE_COORDINATES = (X, Y, Z, T)


class CheckError(AssertionError):
    pass


def assert_equal(label: str, actual: sp.Expr, expected: sp.Expr = sp.Integer(0)) -> None:
    difference = sp.factor(sp.cancel(actual - expected))
    if difference != 0:
        raise CheckError(f"{label}: difference {difference}")


def remainder_mod(expression: sp.Expr, modulus: sp.Expr, parameter: sp.Symbol) -> sp.Expr:
    numerator, denominator = sp.fraction(sp.together(expression))
    modulus_poly = sp.Poly(modulus, parameter)
    numerator_remainder = sp.rem(sp.Poly(numerator, parameter), modulus_poly).as_expr()
    denominator_remainder = sp.rem(sp.Poly(denominator, parameter), modulus_poly).as_expr()
    if sp.gcd(sp.Poly(denominator_remainder, parameter), modulus_poly).degree() != 0:
        raise CheckError(f"denominator {denominator} is not invertible modulo {modulus}")
    return sp.factor(numerator_remainder / denominator_remainder)


def assert_mod(
    label: str,
    actual: sp.Expr,
    expected: sp.Expr,
    modulus: sp.Expr,
    parameter: sp.Symbol,
) -> None:
    if remainder_mod(actual - expected, modulus, parameter) != 0:
        raise CheckError(f"{label}: mismatch modulo {modulus}")


def chart_polynomial(homogeneous: sp.Expr, nonzero_coordinate: sp.Symbol) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    variables = tuple(c for c in PROJECTIVE_COORDINATES if c != nonzero_coordinate)
    return sp.expand(homogeneous.subs(nonzero_coordinate, 1)), variables


def homogeneous_part(
    expression: sp.Expr, variables: tuple[sp.Symbol, ...], degree: int
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), *variables)
    return sp.expand(
        sum(
            coefficient * sp.prod(variable**exponent for variable, exponent in zip(variables, monomial))
            for monomial, coefficient in polynomial.terms()
            if sum(monomial) == degree
        )
    )


def assert_radical_support(
    label: str,
    polynomial: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    radical_generators: tuple[sp.Expr, ...],
    *,
    max_power: int = 12,
) -> None:
    """Verify J(polynomial) has the stated reduced support.

    The supplied ideal is radical in every use below.  We check
    J(polynomial) is contained in it and that a power of every supplied
    generator belongs to J(polynomial); these two inclusions identify the
    radical of the Jacobian ideal.
    """

    jacobian_generators = [polynomial] + [
        sp.diff(polynomial, variable) for variable in variables
    ]
    jacobian_basis = sp.groebner(
        jacobian_generators, *variables, order="lex"
    )
    radical_basis = sp.groebner(
        list(radical_generators), *variables, order="lex"
    )

    for generator in jacobian_generators:
        if radical_basis.reduce(sp.expand(generator))[1] != 0:
            raise CheckError(f"{label}: Jacobian ideal is not contained in stated radical")

    for generator in radical_generators:
        for exponent in range(1, max_power + 1):
            if jacobian_basis.reduce(sp.expand(generator**exponent))[1] == 0:
                break
        else:
            raise CheckError(
                f"{label}: no power <= {max_power} of {generator} lies in the Jacobian ideal"
            )


def check_projective_singular_point(
    label: str,
    homogeneous: sp.Expr,
    point: dict[sp.Symbol, sp.Expr],
    lambda_value: sp.Expr,
    *,
    modulus: sp.Expr | None = None,
    parameter: sp.Symbol | None = None,
) -> None:
    substituted = {**point, lam: lambda_value}
    expressions = (homogeneous,) + tuple(sp.diff(homogeneous, c) for c in PROJECTIVE_COORDINATES)
    for expression in expressions:
        value = sp.factor(expression.subs(substituted))
        if modulus is None:
            assert_equal(f"{label} singular equation", value)
        else:
            assert parameter is not None
            assert_mod(f"{label} singular equation", value, 0, modulus, parameter)


def check_chart_morse_point(
    label: str,
    homogeneous: sp.Expr,
    nonzero_coordinate: sp.Symbol,
    point: dict[sp.Symbol, sp.Expr],
    lambda_value: sp.Expr,
    *,
    modulus: sp.Expr | None = None,
    parameter: sp.Symbol | None = None,
) -> None:
    local, variables = chart_polynomial(homogeneous, nonzero_coordinate)
    local_point = {
        coordinate: sp.cancel(point[coordinate] / point[nonzero_coordinate])
        for coordinate in variables
    }
    determinant = sp.factor(
        sp.hessian(local, variables).det().subs({**local_point, lam: lambda_value})
    )
    if modulus is None:
        if determinant == 0:
            raise CheckError(f"{label}: Hessian vanished in chart {nonzero_coordinate}")
    else:
        assert parameter is not None
        numerator, _denominator = sp.fraction(sp.together(determinant))
        if sp.gcd(sp.Poly(numerator, parameter), sp.Poly(modulus, parameter)).degree() != 0:
            raise CheckError(f"{label}: Hessian degenerates modulo {modulus}")


def logarithmic_hessian(potential: sp.Expr) -> sp.Matrix:
    variables = (x, y, z)

    def theta(expression: sp.Expr, variable: sp.Symbol) -> sp.Expr:
        return sp.factor(variable * sp.diff(expression, variable))

    return sp.Matrix(
        [[theta(theta(potential, column), row) for column in variables] for row in variables]
    )


def validate_b4() -> None:
    potential = (x + 1) ** 2 * (y + 1) ** 2 / (x * y * z) + z
    homogeneous = (X + T) ** 2 * (Y + T) ** 2 + X * Y * Z**2 - lam * X * Y * Z * T
    assert_equal("B4 homogeneity", sum(c * sp.diff(homogeneous, c) for c in PROJECTIVE_COORDINATES), 4 * homogeneous)

    for sign in (1, -1):
        torus_point = {x: 1, y: 1, z: 4 * sign}
        value = 8 * sign
        logarithmic_equations = tuple(variable * sp.diff(potential, variable) for variable in (x, y, z))
        for equation in logarithmic_equations:
            assert_equal("B4 torus critical equation", equation.subs(torus_point))
        assert_equal("B4 torus critical value", potential.subs(torus_point), value)
        determinant = sp.factor(logarithmic_hessian(potential).det().subs(torus_point))
        if determinant == 0:
            raise CheckError("B4 torus Hessian vanished")

        projective_point = {X: 1, Y: 1, Z: 4 * sign, T: 1}
        check_projective_singular_point("B4 ordinary double point", homogeneous, projective_point, value)
        for chart in PROJECTIVE_COORDINATES:
            check_chart_morse_point(
                "B4 ordinary double point", homogeneous, chart, projective_point, value
            )

    central = sp.expand(homogeneous.subs(lam, 0))
    b4_chart_data = (
        (
            "B4 U_T reduced Jacobian support",
            (x + 1) ** 2 * (y + 1) ** 2 + x * y * z**2,
            (x, y, z),
            (z, (x + 1) * (y + 1)),
        ),
        (
            "B4 U_X reduced Jacobian support",
            (1 + t) ** 2 * (y + t) ** 2 + y * z**2,
            (y, z, t),
            (z, (t + 1) * (y + t)),
        ),
        (
            "B4 U_Y reduced Jacobian support",
            (x + t) ** 2 * (1 + t) ** 2 + x * z**2,
            (x, z, t),
            (z, (x + t) * (t + 1)),
        ),
        (
            "B4 U_Z reduced Jacobian support",
            (x + t) ** 2 * (y + t) ** 2 + x * y,
            (x, y, t),
            (x, y, t),
        ),
    )
    for label, local_polynomial, variables, radical_generators in b4_chart_data:
        assert_radical_support(
            label, local_polynomial, variables, radical_generators
        )

    line_x = {X: -T, Z: 0}
    line_y = {Y: -T, Z: 0}
    for label, substitution in (("B4 L_X", line_x), ("B4 L_Y", line_y)):
        for expression in (central,) + tuple(sp.diff(central, c) for c in PROJECTIVE_COORDINATES):
            assert_equal(f"{label} central singular support", expression.subs(substitution))

    crossing = {X: -1, Y: -1, Z: 0, T: 1}
    check_projective_singular_point("B4 line crossing", homogeneous, crossing, 0)
    crossing_form = sp.factor(
        central.subs({X: -1 + u, Y: -1 + v, Z: w, T: 1})
    )
    assert_equal(
        "B4 crossing normal form",
        crossing_form,
        u**2 * v**2 + (-1 + u) * (-1 + v) * w**2,
    )

    pinch_forms = (
        (
            "B4 pinch a_X",
            sp.expand(central.subs({X: -1 + u, Y: v, Z: w, T: 1})),
        ),
        (
            "B4 pinch a_Y",
            sp.expand(central.subs({X: v, Y: -1 + u, Z: w, T: 1})),
        ),
        (
            "B4 pinch b_X",
            sp.expand(central.subs({X: u - v, Y: 1, Z: w, T: v})),
        ),
        (
            "B4 pinch b_Y",
            sp.expand(central.subs({X: 1, Y: u - v, Z: w, T: v})),
        ),
    )
    for label, local_form in pinch_forms:
        assert_equal(
            f"{label} quadratic part",
            homogeneous_part(local_form, (u, v, w), 2),
            u**2,
        )
        residual_cubic = homogeneous_part(local_form.subs(u, 0), (v, w), 3)
        assert_equal(f"{label} D_infinity residual cubic", residual_cubic, -v * w**2)
        if sp.hessian(local_form, (u, v, w)).subs({u: 0, v: 0, w: 0}).rank() != 1:
            raise CheckError(f"{label}: Hessian does not have rank one")

    central_t = sp.expand(central.subs(T, 1))
    for label, point, normals in (
        ("B4 smooth L_X stratum", {X: -1, Y: 1, Z: 0}, (X, Z)),
        ("B4 smooth L_Y stratum", {X: 1, Y: -1, Z: 0}, (Y, Z)),
    ):
        transverse_hessian = sp.hessian(central_t, normals).subs(point)
        if transverse_hessian.det() == 0:
            raise CheckError(f"{label}: transverse Hessian vanished")

    infinity_point = {X: 0, Y: 0, Z: 1, T: 0}
    check_projective_singular_point("B4 boundary A3 point", homogeneous, infinity_point, 0)
    infinity_form = sp.expand(central.subs({X: u, Y: v, Z: 1, T: q}))
    assert_equal(
        "B4 boundary quadratic part",
        homogeneous_part(infinity_form, (u, v, q), 2),
        u * v,
    )
    critical_series = -2 * q**3 + 12 * q**5
    residual = sp.series(
        infinity_form.subs({u: critical_series, v: critical_series}), q, 0, 6
    ).removeO()
    assert_equal("B4 boundary A3 leading residual", residual, q**4)
    infinity_jacobian = sp.groebner(
        [infinity_form]
        + [sp.diff(infinity_form, variable) for variable in (u, v, q)],
        u,
        v,
        q,
        order="lex",
    )
    actual_basis = [sp.factor(polynomial.as_expr()) for polynomial in infinity_jacobian.polys]
    if actual_basis != [u, v, q**3]:
        raise CheckError(f"B4 boundary Jacobian is not A3: {actual_basis}")


def validate_v18() -> None:
    first = X + Y + Z
    second = first * T**2 + (X * Y + X * Z + Y * Z) * T + X * Y * Z
    homogeneous = sp.expand(first * second - lam * X * Y * Z * T)
    assert_equal("V18 homogeneity", sum(c * sp.diff(homogeneous, c) for c in PROJECTIVE_COORDINATES), 4 * homogeneous)

    potential = (x + y + z) * (x + y + z + x * y + x * z + y * z + x * y * z) / (x * y * z)
    modulus = 3 * a**2 - 1
    torus_point = {x: 1 / a, y: 1 / a, z: 1 / a}
    for variable in (x, y, z):
        assert_mod(
            "V18 torus critical equation",
            (variable * sp.diff(potential, variable)).subs(torus_point),
            0,
            modulus,
            a,
        )
    assert_mod("V18 torus critical value", potential.subs(torus_point), 18 * a + 9, modulus, a)
    determinant = sp.factor(logarithmic_hessian(potential).det().subs(torus_point))
    numerator, _denominator = sp.fraction(sp.together(determinant))
    if sp.gcd(sp.Poly(numerator, a), sp.Poly(modulus, a)).degree() != 0:
        raise CheckError("V18 torus Hessian degenerates")

    projective_point = {X: 1, Y: 1, Z: 1, T: a}
    check_projective_singular_point(
        "V18 ordinary double point",
        homogeneous,
        projective_point,
        18 * a + 9,
        modulus=modulus,
        parameter=a,
    )
    for chart in PROJECTIVE_COORDINATES:
        check_chart_morse_point(
            "V18 ordinary double point",
            homogeneous,
            chart,
            projective_point,
            18 * a + 9,
            modulus=modulus,
            parameter=a,
        )

    central_curve = T * (X**2 + X * Y + Y**2) + X * Y * (X + Y)
    assert_equal(
        "V18 central intersection curve",
        second.subs(Z, -X - Y),
        -central_curve,
    )
    factorization = sp.factor_list(central_curve, extension=sp.I)[1]
    if len(factorization) != 1 or factorization[0][1] != 1:
        raise CheckError("V18 central cubic unexpectedly factors")
    affine_node_equation = sp.expand(central_curve.subs(T, 1))
    quadratic_part = X**2 + X * Y + Y**2
    assert_equal(
        "V18 nodal cubic quadratic part",
        homogeneous_part(affine_node_equation, (X, Y), 2),
        quadratic_part,
    )
    if sp.hessian(quadratic_part, (X, Y)).det() == 0:
        raise CheckError("V18 central cubic does not have an ordinary node")

    tau, exceptional_parameter = sp.symbols("tau_v18 exceptional_v18")
    blown_up_node = sp.expand(
        affine_node_equation.subs(
            {X: tau * exceptional_parameter, Y: exceptional_parameter}
        )
        / exceptional_parameter**2
    )
    assert_equal(
        "V18 strict transform at the nodal blowup",
        blown_up_node,
        tau**2
        + tau
        + 1
        + tau * (tau + 1) * exceptional_parameter,
    )
    exceptional_intersection = sp.Poly(
        blown_up_node.subs(exceptional_parameter, 0), tau
    )
    if exceptional_intersection.discriminant() == 0:
        raise CheckError("V18 nodal blowup does not have two distinct tangent points")

    a_t = x + y + z
    b_t = a_t + x * y + x * z + y * z + x * y * z
    a_x = 1 + y + z
    b_x = a_x * t**2 + (y + z + y * z) * t + y * z
    a_y = x + 1 + z
    b_y = a_y * t**2 + (x + z + x * z) * t + x * z
    a_z = x + y + 1
    b_z = a_z * t**2 + (x + y + x * y) * t + x * y
    v18_chart_data = (
        (
            "V18 U_T reduced Jacobian support",
            a_t * b_t,
            (x, y, z),
            (a_t, b_t),
        ),
        (
            "V18 U_X reduced Jacobian support",
            a_x * b_x,
            (y, z, t),
            tuple(factor * coordinate for factor in (a_x, b_x) for coordinate in (y, z, t)),
        ),
        (
            "V18 U_Y reduced Jacobian support",
            a_y * b_y,
            (x, z, t),
            tuple(factor * coordinate for factor in (a_y, b_y) for coordinate in (x, z, t)),
        ),
        (
            "V18 U_Z reduced Jacobian support",
            a_z * b_z,
            (x, y, t),
            tuple(factor * coordinate for factor in (a_z, b_z) for coordinate in (x, y, t)),
        ),
    )
    for label, local_polynomial, variables, radical_generators in v18_chart_data:
        assert_radical_support(
            label, local_polynomial, variables, radical_generators
        )

    torus_central_point = {x: sp.Rational(-3, 2), y: sp.Rational(-3, 2), z: 3}
    for variable in (x, y, z):
        assert_equal(
            "V18 standard-torus central critical equation",
            (variable * sp.diff(potential, variable)).subs(torus_central_point),
        )
    assert_equal("V18 standard-torus central critical value", potential.subs(torus_central_point))
    node = {X: 0, Y: 0, Z: 0, T: 1}
    check_projective_singular_point("V18 central node", homogeneous, node, 0)

    # At the node, use r=A as one coordinate and eliminate Z=r-X-Y.
    # Writing B=r+h gives AB=r(r+h)=(r+h/2)^2-(h/2)^2.  The
    # quadratic part of h|_{r=0} is nondegenerate, hence the formal type is
    # w^2+(uv)^2 after the splitting and formal Morse lemmas.
    r = q
    z_substitution = r - X - Y
    local_second = sp.expand(second.subs({Z: z_substitution, T: 1}))
    h = sp.expand(local_second - r)
    local_central = sp.expand(central_curve.subs(T, 1))
    assert_equal(
        "V18 node difference function on r=0",
        h.subs(r, 0),
        -local_central,
    )
    assert_equal(
        "V18 node square completion",
        r * local_second,
        (r + h / 2) ** 2 - (h / 2) ** 2,
    )
    node_quadratic = homogeneous_part(h.subs(r, 0), (X, Y), 2)
    if sp.hessian(node_quadratic, (X, Y)).det() == 0:
        raise CheckError("V18 node residual quadratic form degenerated")

    # The raw central quartic has three further isolated boundary
    # singularities.  In the chart where the nonzero coordinate is set to
    # one, the Hessian has corank one and its restriction to the kernel starts
    # with a nonzero cubic term, so each point is A2.
    boundary_points = (
        (X, {X: 1, Y: 0, Z: 0, T: 0}),
        (Y, {X: 0, Y: 1, Z: 0, T: 0}),
        (Z, {X: 0, Y: 0, Z: 1, T: 0}),
    )
    s = sp.symbols("s_v18_boundary")
    for chart, point in boundary_points:
        check_projective_singular_point(
            "V18 raw boundary A2 point", homogeneous, point, 0
        )
        local, variables = chart_polynomial(homogeneous.subs(lam, 0), chart)
        origin = {variable: 0 for variable in variables}
        hessian = sp.hessian(local, variables).subs(origin)
        if hessian.rank() != 2 or len(hessian.nullspace()) != 1:
            raise CheckError("V18 raw boundary point does not have Hessian corank one")
        kernel = hessian.nullspace()[0]
        kernel_substitution = {
            variable: s * kernel[index] for index, variable in enumerate(variables)
        }
        restriction = sp.expand(local.subs(kernel_substitution))
        leading = sp.expand(restriction).coeff(s, 3)
        if leading == 0:
            raise CheckError("V18 raw boundary point has no cubic residual term")

    # The other three boundary points listed in the text lie on the smooth
    # part of C0; the fourth special point is the node p0 already checked.
    smooth_boundary_points = (
        {X: 1, Y: -1, Z: 0, T: 0},
        {X: 1, Y: 0, Z: -1, T: 0},
        {X: 0, Y: 1, Z: -1, T: 0},
    )
    jacobian_pair = sp.Matrix(
        [
            [sp.diff(first, coordinate) for coordinate in PROJECTIVE_COORDINATES],
            [sp.diff(second, coordinate) for coordinate in PROJECTIVE_COORDINATES],
        ]
    )
    for point in smooth_boundary_points:
        assert_equal("V18 smooth boundary point first equation", first.subs(point))
        assert_equal("V18 smooth boundary point second equation", second.subs(point))
        if jacobian_pair.subs(point).rank() != 2:
            raise CheckError("V18 boundary point is not a smooth point of C0")


def validate_v6_extreme() -> None:
    local = (x + 1) ** 2 * (y + z + 1) ** 3 - lam * x * y * z
    point = {x: 1, y: 1, z: 1, lam: 108}
    assert_equal("V6 compact node equation", local.subs(point))
    for variable in (x, y, z):
        assert_equal(
            "V6 compact node first derivative", sp.diff(local, variable).subs(point)
        )
    hessian = sp.hessian(local, (x, y, z)).subs(point)
    expected_hessian = sp.Matrix(
        [[54, 0, 0], [0, 72, -36], [0, -36, 72]]
    )
    if hessian != expected_hessian:
        raise CheckError(f"V6 compact Hessian mismatch: {hessian}")
    assert_equal("V6 compact Hessian determinant", hessian.det(), 2**5 * 3**8)


def validate_v22_extreme() -> None:
    potential = (
        x * y / z
        + y / z
        + x / z
        + x
        + y
        + 1 / z
        + 4
        + 1 / x
        + 1 / y
        + z
        + 1 / (x * y)
        + z / x
        + z / y
        + z / (x * y)
    )
    homogeneous = (
        X**2 * Y**2
        + X * Y**2 * T
        + X**2 * Y * T
        + X**2 * Y * Z
        + X * Y**2 * Z
        + X * Y * T**2
        + 4 * X * Y * Z * T
        + Y * Z * T**2
        + X * Z * T**2
        + X * Y * Z**2
        + Z * T**3
        + Y * Z**2 * T
        + X * Z**2 * T
        + Z**2 * T**2
        - lam * X * Y * Z * T
    )
    assert_equal(
        "V22 direct homogenization",
        homogeneous.subs({X: x, Y: y, Z: z, T: 1}) / (x * y * z),
        potential - lam,
    )
    assert_equal(
        "V22 quartic homogeneity",
        sum(c * sp.diff(homogeneous, c) for c in PROJECTIVE_COORDINATES),
        4 * homogeneous,
    )

    modulus = a**3 + 2 * a**2 - 2
    lambda_value = 3 * a**2 + 8 * a + 8
    projective_point = {X: 1, Y: 1, Z: 1, T: a}
    check_projective_singular_point(
        "V22 non-base ordinary double point",
        homogeneous,
        projective_point,
        lambda_value,
        modulus=modulus,
        parameter=a,
    )
    local_x, variables = chart_polynomial(homogeneous, X)
    local_point = {Y: 1, Z: 1, T: a, lam: lambda_value}
    determinant = sp.factor(sp.hessian(local_x, variables).det().subs(local_point))
    assert_mod(
        "V22 non-base Hessian determinant",
        determinant,
        2 * (14 * a**2 + 39 * a + 32),
        modulus,
        a,
    )
    resultant = sp.resultant(modulus, 14 * a**2 + 39 * a + 32, a)
    assert_equal("V22 Hessian resultant", resultant, 22)
    if sp.discriminant(modulus, a) == 0:
        raise CheckError("V22 node polynomial has a repeated root")

    q22, r22, t22 = sp.symbols("q22 r22 t22")
    total_transform = sp.expand(
        homogeneous.subs(
            {X: 1, Y: q22, Z: t22 * r22 - q22, T: t22}
        )
    )
    strict_transform = sp.factor(sp.cancel(total_transform / t22))
    expected_strict_transform = (
        lam * q22**2
        + q22**3
        - q22**2 * r22
        - 2 * q22**2
        + q22 * r22
        + q22
        + q22 * r22 * t22 * (-lam - 2 * q22 + r22 + 2)
        + t22**2
        * (q22 * r22**2 - q22 * r22 - q22 + r22**2 + r22)
        + r22 * (r22 + 1) * t22**3
    )
    assert_equal(
        "V22 boundary strict transform", strict_transform, expected_strict_transform
    )
    boundary_point = {q22: 1, r22: 0, t22: 0, lam: 0}
    assert_equal("V22 boundary node equation", strict_transform.subs(boundary_point))
    for variable in (q22, r22, t22):
        assert_equal(
            "V22 boundary node first derivative",
            sp.diff(strict_transform, variable).subs(boundary_point),
        )
    boundary_hessian = sp.hessian(
        strict_transform, (q22, r22, t22)
    ).subs(boundary_point)
    expected_boundary_hessian = sp.Matrix([[2, -1, 0], [-1, 0, 0], [0, 0, -2]])
    if boundary_hessian != expected_boundary_hessian:
        raise CheckError(f"V22 boundary Hessian mismatch: {boundary_hessian}")
    assert_equal("V22 boundary Hessian determinant", boundary_hessian.det(), 2)
    transformed_base_generator = q22 * (t22 * r22 - q22)
    assert_equal(
        "V22 boundary node leaves the base locus",
        transformed_base_generator.subs(boundary_point),
        -1,
    )


def main() -> int:
    validate_b4()
    print("PASS B_4 five-chart compactified audit")
    validate_v18()
    print("PASS V_18 five-chart compactified audit")
    validate_v6_extreme()
    print("PASS V_6 compactified node and Hessian audit")
    validate_v22_extreme()
    print("PASS V_22 three non-base nodes and boundary blowup audit")
    print("PASS chart overlap count: each noncentral ordinary double point is one global point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())