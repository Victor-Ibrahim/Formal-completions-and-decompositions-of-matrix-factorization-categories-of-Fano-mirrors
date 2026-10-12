#!/usr/bin/env python3
"""
Build an auditable Mori--Mukai computational-certificate supplement.

Input:  computations/mori_mukai_validation.py
Output: supplementary-morimukai-certificates.tex

The generated supplement is intentionally plain: each Mori--Mukai label gets a
monospaced certificate block grouped into identities, saturated ideals, point and
Hessian checks, parametric branches, and normal forms/expansions.  The data are
read directly from the CASES dictionary used by the validation script, so the
supplement stays synchronized with the executable checks.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List

OUT_TEX = Path("supplementary-morimukai-certificates.tex")
VALIDATION_SCRIPT = Path("computations/mori_mukai_validation.py")
WRAP_WIDTH = 132


def load_validation_module():
    spec = importlib.util.spec_from_file_location("mori_mukai_validation", VALIDATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {VALIDATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mori_mukai_validation"] = module
    spec.loader.exec_module(module)
    return module


mmv = load_validation_module()


def label_tex(label: str) -> str:
    _prefix, rank, number = label.split("_")
    return f"MM_{{{rank}-{number}}}"


def expr_text(expr) -> str:
    return str(expr)


def tuple_text(items: Iterable) -> str:
    return ", ".join(expr_text(item) for item in items) if items else "none"


def mapping_text(mapping) -> str:
    if not mapping:
        return "none"
    ordered = sorted(mapping.items(), key=lambda item: str(item[0]))
    return ", ".join(f"{symbol} = {expr_text(value)}" for symbol, value in ordered)


def matrix_text(matrix) -> str:
    if matrix is None:
        return "none"
    return expr_text(matrix)


def add_wrapped(lines: List[str], line: str = "") -> None:
    if not line:
        lines.append("")
        return
    indent = len(line) - len(line.lstrip(" "))
    prefix = " " * indent
    wrapped = textwrap.wrap(
        line,
        width=WRAP_WIDTH,
        subsequent_indent=prefix + "  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines.extend(wrapped or [line])


def certificate_block(name: str, case) -> str:
    lines: List[str] = []
    add_wrapped(lines, f"Label: {name}")
    add_wrapped(lines, f"Notes: {case.notes or 'none'}")
    add_wrapped(lines)

    add_wrapped(lines, "Identities / compact forms:")
    if case.identities:
        for identity_label, identity in case.identities:
            add_wrapped(lines, f"  - {identity_label}: {expr_text(identity)} == 0")
    else:
        add_wrapped(lines, "  none")
    add_wrapped(lines)

    add_wrapped(lines, "Saturated critical ideals:")
    if case.saturation_checks:
        for check in case.saturation_checks:
            add_wrapped(lines, f"  - {check.label}")
            add_wrapped(lines, f"    saturation element: {expr_text(check.by)}")
            add_wrapped(lines, "    expected generators:")
            for generator in check.expected_generators:
                add_wrapped(lines, f"      * {expr_text(generator)}")
    else:
        add_wrapped(lines, "  none")
    add_wrapped(lines)

    add_wrapped(lines, "Point and Hessian certificates:")
    any_point = bool(case.point_checks or case.parametric_point_checks)
    if case.point_checks:
        for point in case.point_checks:
            add_wrapped(lines, f"  - point: {point.label}")
            add_wrapped(lines, f"    coordinates: {mapping_text(point.point)}")
            add_wrapped(lines, f"    value: {expr_text(point.value) if point.value is not None else 'not asserted'}")
            add_wrapped(lines, f"    ordinary Hessian det: {expr_text(point.ordinary_hessian_det) if point.ordinary_hessian_det is not None else 'not asserted'}")
            add_wrapped(lines, f"    log Hessian det: {expr_text(point.log_hessian_det) if point.log_hessian_det is not None else 'not asserted'}")
            add_wrapped(lines, f"    require nonzero ordinary Hessian: {point.require_nonzero_ordinary}")
            add_wrapped(lines, f"    require nonzero log Hessian: {point.require_nonzero_log}")
    if case.parametric_point_checks:
        for point in case.parametric_point_checks:
            add_wrapped(lines, f"  - parametric point branch: {point.label}")
            add_wrapped(lines, f"    parameter: {point.parameter}")
            add_wrapped(lines, f"    minimal polynomial: {expr_text(point.minimal_polynomial)}")
            add_wrapped(lines, f"    coordinates: {mapping_text(point.point)}")
            add_wrapped(lines, f"    value: {expr_text(point.value) if point.value is not None else 'not asserted'}")
            add_wrapped(lines, f"    ordinary Hessian det: {expr_text(point.ordinary_hessian_det) if point.ordinary_hessian_det is not None else 'not asserted'}")
            add_wrapped(lines, f"    log Hessian det: {expr_text(point.log_hessian_det) if point.log_hessian_det is not None else 'not asserted'}")
            add_wrapped(lines, f"    coprime/nonvanishing tests: {tuple_text(point.coprime_tests)}")
    if not any_point:
        add_wrapped(lines, "  none")
    add_wrapped(lines)

    add_wrapped(lines, "Parametric branches / positive-dimensional components:")
    if case.branch_checks:
        for branch in case.branch_checks:
            add_wrapped(lines, f"  - branch: {branch.label}")
            add_wrapped(lines, f"    parameter: {branch.parameter}")
            add_wrapped(lines, f"    substitution: {mapping_text(branch.substitution)}")
            add_wrapped(lines, f"    exclusions: {tuple_text(branch.exclusions)}")
            add_wrapped(lines, f"    value: {expr_text(branch.value) if branch.value is not None else 'not asserted'}")
            add_wrapped(lines, f"    log Hessian matrix: {matrix_text(branch.log_hessian_matrix)}")
    else:
        add_wrapped(lines, "  none")
    add_wrapped(lines)

    add_wrapped(lines, "Normal forms and formal expansions:")
    any_normal = bool(case.normal_form_checks or case.expansion_checks)
    if case.normal_form_checks:
        for normal in case.normal_form_checks:
            add_wrapped(lines, f"  - normal form: {normal.label}")
            add_wrapped(lines, f"    checked identity: W - ({expr_text(normal.critical_value)}) = ({expr_text(normal.extra_term)}) + ({expr_text(normal.unit)})*({expr_text(normal.residual)})^{normal.exponent}")
            add_wrapped(lines, f"    residual: {expr_text(normal.residual)}")
            add_wrapped(lines, f"    exponent: {normal.exponent}")
            add_wrapped(lines, f"    unit: {expr_text(normal.unit)}")
            add_wrapped(lines, f"    extra term: {expr_text(normal.extra_term)}")
    if case.expansion_checks:
        for expansion in case.expansion_checks:
            add_wrapped(lines, f"  - formal expansion: {expansion.label}")
            add_wrapped(lines, f"    substitution: {mapping_text(expansion.substitution)}")
            add_wrapped(lines, f"    expected expression through degree < {expansion.order}: {expr_text(expansion.expected)}")
            add_wrapped(lines, f"    normal variables: {tuple_text(expansion.normal_variables)}")
    if not any_normal:
        add_wrapped(lines, "  none")

    return "\n".join(lines)


def collect_counts(cases) -> dict[str, int]:
    return {
        "labels": len(cases),
        "saturation_checks": sum(len(case.saturation_checks) for case in cases.values()),
        "point_checks": sum(len(case.point_checks) for case in cases.values()),
        "parametric_point_checks": sum(len(case.parametric_point_checks) for case in cases.values()),
        "branch_checks": sum(len(case.branch_checks) for case in cases.values()),
        "normal_form_checks": sum(len(case.normal_form_checks) for case in cases.values()),
        "expansion_checks": sum(len(case.expansion_checks) for case in cases.values()),
        "identity_checks": sum(len(case.identities) for case in cases.values()),
    }


def build_document() -> str:
    cases = {name: mmv.CASES[name] for name in sorted(mmv.CASES, key=mmv.case_sort_key)}
    counts = collect_counts(cases)
    body_parts = []
    for name, case in cases.items():
        body_parts.append(f"\\subsection*{{$ {label_tex(name)} $}}")
        body_parts.append("\\begin{lstlisting}")
        body_parts.append(certificate_block(name, case))
        body_parts.append("\\end{lstlisting}")

    body = "\n\n".join(body_parts)
    return rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{listings}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black]{{hyperref}}
\lstdefinestyle{{certificates}}{{
  basicstyle=\ttfamily\tiny,
  columns=fullflexible,
  breaklines=true,
  breakatwhitespace=false,
  keepspaces=true,
  showstringspaces=false,
  frame=single,
  framerule=0.2pt,
  xleftmargin=0.5em,
  xrightmargin=0.5em
}}
\lstset{{style=certificates}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.35em}}
\sloppy

\title{{Auditable Computational Certificates for Mori--Mukai Landau--Ginzburg Models}}
\author{{Generated from \texttt{{computations/mori\_mukai\_validation.py}}}}
\date{{11 de agosto de 2026}}

\begin{{document}}
\maketitle

\section*{{How to Read This Supplement}}
The source is the executable validation dictionary \texttt{{CASES}} in
\path{{computations/mori_mukai_validation.py}}. Each Mori--Mukai label is listed separately.
Within each label, the certificates are grouped as identities/compact forms, saturated critical ideals,
point and Hessian checks, parametric branches, and normal forms or formal expansions.  Regenerate this
supplement with
\[
\texttt{{python computations/build\_morimukai\_certificates.py}}
\]
Running the companion symbolic-validation command
\[
\texttt{{python computations/mori\_mukai\_validation.py --all}}
\]
recomputes the symbolic checks represented here.

\section*{{Summary}}
\begin{{itemize}}
\item Mori--Mukai labels: {counts['labels']}.
\item Compact-form/identity checks: {counts['identity_checks']}.
\item Saturated-ideal checks: {counts['saturation_checks']}.
\item Explicit point checks: {counts['point_checks']}.
\item Parametric isolated-point checks: {counts['parametric_point_checks']}.
\item Parametric branch/component checks: {counts['branch_checks']}.
\item Exact normal-form checks: {counts['normal_form_checks']}.
\item Formal expansion checks: {counts['expansion_checks']}.
\end{{itemize}}

\section*{{Certificates by Label}}
{body}

\end{{document}}
"""


def main() -> int:
    OUT_TEX.write_text(build_document(), encoding="utf-8")
    print(f"Wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())