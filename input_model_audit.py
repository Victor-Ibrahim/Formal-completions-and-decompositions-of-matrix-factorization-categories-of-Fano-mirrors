#!/usr/bin/env python3
"""Audit source-level invariants before running model-internal algebra checks."""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import sympy as sp

from mori_mukai_validation import CASES, x, y, z


VARS = (x, y, z)
NON_MINKOWSKI = {"MM_2_1", "MM_2_2", "MM_2_3", "MM_9_1", "MM_10_1"}
MM_3_7_CCGK = x * y / z + x + x / z + x / y + y + z + 2 / y + y / x + 2 / x + 1 / (x * y)
MM_3_7_PERIOD = (1, 0, 10, 48, 438, 3720, 33940)
MM_4_13_CCGK = x * y + x * y / z + x + y + z + 1 / y + 1 / x


def exponent_coefficients(expr: sp.Expr) -> dict[tuple[int, int, int], sp.Expr]:
    result: dict[tuple[int, int, int], sp.Expr] = {}
    for term in sp.expand(expr).as_ordered_terms():
        powers = term.as_powers_dict()
        exponent = tuple(int(powers.get(variable, 0)) for variable in VARS)
        monomial = sp.prod(variable**power for variable, power in zip(VARS, exponent))
        coefficient = sp.simplify(term / monomial)
        result[exponent] = sp.simplify(result.get(exponent, 0) + coefficient)
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient != 0}


def primitive(vector: tuple[int, int, int]) -> tuple[int, int, int]:
    divisor = math.gcd(math.gcd(abs(vector[0]), abs(vector[1])), abs(vector[2]))
    return tuple(entry // divisor for entry in vector)


def facet_distances(expr: sp.Expr) -> set[int]:
    points = tuple(exponent_coefficients(expr))
    facets: set[tuple[tuple[int, int, int], int]] = set()
    for first, second, third in itertools.combinations(points, 3):
        edge_one = tuple(second[index] - first[index] for index in range(3))
        edge_two = tuple(third[index] - first[index] for index in range(3))
        normal = (
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        )
        if normal == (0, 0, 0):
            continue
        normal = primitive(normal)
        level = sum(normal[index] * first[index] for index in range(3))
        values = [sum(normal[index] * point[index] for index in range(3)) for point in points]
        if all(value <= level for value in values):
            pass
        elif all(value >= level for value in values):
            normal = tuple(-entry for entry in normal)
            level = -level
        else:
            continue
        if level <= 0:
            raise AssertionError("the origin is not strictly inside the Newton polytope")
        facets.add((normal, level))
    return {level for _normal, level in facets}


def constant_term_period(expr: sp.Expr, order: int) -> tuple[sp.Expr, ...]:
    terms = exponent_coefficients(expr)
    state = {(0, 0, 0): sp.Integer(1)}
    coefficients = [sp.Integer(1)]
    for _degree in range(1, order + 1):
        next_state: dict[tuple[int, int, int], sp.Expr] = {}
        for left_exp, left_coeff in state.items():
            for right_exp, right_coeff in terms.items():
                exponent = tuple(left_exp[index] + right_exp[index] for index in range(3))
                next_state[exponent] = next_state.get(exponent, 0) + left_coeff * right_coeff
        state = next_state
        coefficients.append(sp.expand(state.get((0, 0, 0), 0)))
    return tuple(coefficients)


def audit_reflexivity() -> None:
    for name, case in CASES.items():
        if name in NON_MINKOWSKI:
            continue
        distances = facet_distances(case.potential)
        if distances != {1}:
            raise AssertionError(f"{name}: non-reflexive Newton polytope, facet distances {sorted(distances)}")
    print("PASS reflexivity for all 83 non-exceptional Mori--Mukai inputs")


def audit_mm_3_7_source_and_period() -> None:
    if sp.simplify(CASES["MM_3_7"].potential - MM_3_7_CCGK) != 0:
        raise AssertionError("MM_3_7 does not match CCGK Appendix A")
    actual = constant_term_period(MM_3_7_CCGK, len(MM_3_7_PERIOD) - 1)
    if actual != MM_3_7_PERIOD:
        raise AssertionError(f"MM_3_7 period mismatch: {actual} != {MM_3_7_PERIOD}")
    print("PASS MM_3_7 CCGK source polynomial and period through degree 6")


def audit_rank_four_source_labels() -> None:
    if sp.simplify(CASES["MM_4_13"].potential - MM_4_13_CCGK) != 0:
        raise AssertionError("MM_4_13 does not match CCGK Appendix A")
    seen: dict[str, str] = {}
    for name, case in CASES.items():
        key = sp.srepr(sp.expand(case.potential))
        if key in seen:
            raise AssertionError(f"duplicate Mori--Mukai inputs: {seen[key]} and {name}")
        seen[key] = name
    print("PASS MM_4_13 CCGK source polynomial and absence of duplicate inputs")


def audit_fano_count_benchmark(path: Path = Path("main.tex")) -> None:
    text = path.read_text(encoding="utf-8")
    table = text.split("label = {tab:fano_threefold_decompositions}", 1)[1].split("\\end{longtblr}", 1)[0]
    rows = re.findall(r"^\$(.+?)\$\s*&\s*(\d+)\s*&", table, flags=re.M)
    for label, count_text in rows:
        count = int(count_text)
        match = re.fullmatch(r"MM_\{(\d+)-(\d+)\}", label)
        picard_rank = int(match.group(1)) if match else 1
        benchmark = 2 + 2 * picard_rank
        if count > benchmark:
            raise AssertionError(f"{label}: open count {count} exceeds benchmark {benchmark}")
    print(f"PASS open-count/cohomological-benchmark consistency for {len(rows)} Fano-threefold rows")


def audit_del_pezzo_count_benchmark(path: Path = Path("main.tex")) -> None:
    text = path.read_text(encoding="utf-8")
    table = text.split("label = {tab:dp_decompositions}", 1)[1].split("\\end{longtblr}", 1)[0]
    rows = re.findall(r"^\$(.+?)\$\s*&\s*(\d+)\s*&\s*(\d+)\s*&", table, flags=re.M)
    for label, count_text, benchmark_text in rows:
        count = int(count_text)
        benchmark = int(benchmark_text)
        if count > benchmark:
            raise AssertionError(f"{label}: open count {count} exceeds benchmark {benchmark}")
    print(f"PASS open-count/cohomological-benchmark consistency for {len(rows)} del Pezzo rows")


def audit_rank_one_compactified_comparison(path: Path = Path("main.tex")) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("Threefolds & Open isolated Morse points")
    table = text[start:text.index("\\end{tblr}", start)]
    rows = re.findall(r"^(\$.+?\$)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\$", table, flags=re.M)
    expected = {
        "$V_2,V_4,V_6,V_8$": (1, 1, 0),
        "$V_{10},V_{12},V_{14},V_{16},V_{18}$": (2, 2, 0),
        "$B_1,B_2,B_3,B_4$": (2, 2, 0),
        "$B_5,\\mathbb P^3$": (4, 4, 0),
        "$V_{22}$": (3, 4, 1),
        "$Q$": (3, 4, 1),
    }
    actual = {label: tuple(map(int, values)) for label, *values in rows}
    if actual != expected:
        raise AssertionError(f"rank-one compactified comparison mismatch: {actual}")
    equality_count = sum(open_count == compact_count for open_count, compact_count, _difference in actual.values())
    if equality_count != 4:
        raise AssertionError(f"grouped comparison has {equality_count} equality rows instead of 4")
    family_equality_count = sum(
        label.count(",") + 1
        for label, (open_count, compact_count, _difference) in actual.items()
        if open_count == compact_count
    )
    if family_equality_count != 15:
        raise AssertionError("rank-one family equality count is not 15")
    deficit_count = sum(
        label.count(",") + 1
        for label, (open_count, compact_count, _difference) in actual.items()
        if open_count != compact_count
    )
    if deficit_count != 2:
        raise AssertionError("rank-one family deficit count is not 2")
    print("PASS rank-one open/compactified isolated-point comparison (15 exact, 2 deficit-one)")


def main() -> int:
    audit_mm_3_7_source_and_period()
    audit_rank_four_source_labels()
    audit_reflexivity()
    audit_fano_count_benchmark()
    audit_del_pezzo_count_benchmark()
    audit_rank_one_compactified_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())