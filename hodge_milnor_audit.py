#!/usr/bin/env python3
"""Audit the Hodge/Milnor statistics of the 105 displayed Fano models.

The Hodge numbers follow the Iskovskikh--Mori--Mukai numbering used by
Fanography and by the tables in Belmans--Fatighenti--Tanturri.  The script
parses the decomposition table in ``main.tex``.  A row is classified as
non-isolated when its displayed decomposition contains a positive-dimensional
or connected formal contribution.  Otherwise the table consists entirely of
``Q_3`` terms, so its ``Count`` is the total Milnor number of its Morse points.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


H12 = {
    "1-1": 52, "1-2": 30, "1-3": 20, "1-4": 14, "1-5": 10,
    "1-6": 7, "1-7": 5, "1-8": 3, "1-9": 2, "1-10": 0,
    "1-11": 21, "1-12": 10, "1-13": 5, "1-14": 2,
    "1-15": 0, "1-16": 0, "1-17": 0,
    "2-1": 22, "2-2": 20, "2-3": 11, "2-4": 10, "2-5": 6,
    "2-6": 9, "2-7": 5, "2-8": 9, "2-9": 5, "2-10": 3,
    "2-11": 5, "2-12": 3, "2-13": 2, "2-14": 1, "2-15": 4,
    "2-16": 2, "2-17": 1, "2-18": 2, "2-19": 2, "2-20": 0,
    "2-21": 0, "2-22": 0, "2-23": 1, "2-24": 0, "2-25": 1,
    "2-26": 0, "2-27": 0, "2-28": 1, "2-29": 0, "2-30": 0,
    "2-31": 0, "2-32": 0, "2-33": 0, "2-34": 0, "2-35": 0,
    "2-36": 0,
    "3-1": 8, "3-2": 3, "3-3": 3, "3-4": 2, "3-5": 0,
    "3-6": 1, "3-7": 1, "3-8": 0, "3-9": 3, "3-10": 0,
    "3-11": 1, "3-12": 0, "3-13": 0, "3-14": 1, "3-15": 0,
    "3-16": 0, "3-17": 0, "3-18": 0, "3-19": 0, "3-20": 0,
    "3-21": 0, "3-22": 0, "3-23": 0, "3-24": 0, "3-25": 0,
    "3-26": 0, "3-27": 0, "3-28": 0, "3-29": 0, "3-30": 0,
    "3-31": 0,
    "4-1": 1, "4-2": 1, "4-3": 0, "4-4": 0, "4-5": 0,
    "4-6": 0, "4-7": 0, "4-8": 0, "4-9": 0, "4-10": 0,
    "4-11": 0, "4-12": 0, "4-13": 0,
    "5-1": 0, "5-2": 0, "5-3": 0,
    "6-1": 0, "7-1": 0, "8-1": 0, "9-1": 0, "10-1": 0,
}


RANK_ONE_IDS = {
    r"\mathbb{P}^3": "1-17",
    r"Q\subset\mathbb{P}^4": "1-16",
    "B_5": "1-15", "B_4": "1-14", "B_3": "1-13",
    "B_2": "1-12", "B_1": "1-11", "V_{22}": "1-10",
    "V_{18}": "1-9", "V_{16}": "1-8", "V_{14}": "1-7",
    "V_{12}": "1-6", "V_{10}": "1-5", "V_8": "1-4",
    "V_6": "1-3", "V_4": "1-2", "V_2": "1-1",
}


# Cases for which generic semisimplicity of small quantum cohomology is
# recorded in the Fanography Dubrovin table.  This is comparison data only:
# it is not used to identify the displayed q=1 Laurent polynomial with a
# generic-q mirror.
KNOWN_SMALL_QH_SEMISIMPLE = {
    "1-10", "1-15", "1-16", "1-17",
    "2-21", "2-22", "2-24", "2-26", "2-27", "2-29", "2-30",
    "2-31", "2-32", "2-33", "2-34", "2-35", "2-36",
    "3-10", "3-12", "3-15", "3-17", "3-18", "3-20", "3-24",
    "3-25", "3-26", "3-27", "3-28", "3-29", "3-30", "3-31",
    "4-9", "4-10", "4-11", "4-12", "5-2", "5-3",
    "6-1", "7-1", "8-1", "9-1", "10-1",
}


# The companion computations give an exact repeated-factor expression for a
# singular fiber in precisely these h^{1,2}>0 models.  For example, the list
# contains fibers of the form F^2Q, xU^m (m>1), or AG^2.  Thus the fiber is
# non-reduced as a hypersurface in the algebraic torus.  This is a statement
# about the displayed specialization, not a birationality criterion.
NONREDUCED_H12_POSITIVE = {
    "1-1", "1-2", "1-3", "1-4", "1-5", "1-11", "1-12",
    "2-1", "2-2", "2-3", "2-8",
}


NONISOLATED_MARKERS = (
    r"\MBcat", r"\Aper", r"\Dper", r"\QAper", r"\QAtwist",
    r"\mathcal C",
)


@dataclass(frozen=True)
class Row:
    identifier: str
    label: str
    picard_rank: int
    h12: int
    count: int
    even_dimension: int
    decomposition: str
    nonisolated: bool

    @property
    def euler_number(self) -> int:
        return 2 + 2 * self.picard_rank - 2 * self.h12

    @property
    def milnor_total(self) -> int | None:
        return None if self.nonisolated else self.count


def identifier(label: str) -> str:
    if label in RANK_ONE_IDS:
        return RANK_ONE_IDS[label]
    match = re.fullmatch(r"MM_\{(\d+)-(\d+)\}", label)
    if match is None:
        raise AssertionError(f"unrecognized Fano label: {label}")
    return f"{match.group(1)}-{match.group(2)}"


def parse_rows(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    table = text.split("label = {tab:fano_threefold_decompositions}", 1)[1]
    table = table.split(r"\end{longtblr}", 1)[0]
    pattern = re.compile(
        r"^\$(.+?)\$\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(.*?)\\\\$",
        flags=re.M,
    )
    rows = []
    for label, count_text, even_text, decomposition in pattern.findall(table):
        model_id = identifier(label)
        picard_rank = int(model_id.split("-", 1)[0])
        row = Row(
            identifier=model_id,
            label=label,
            picard_rank=picard_rank,
            h12=H12[model_id],
            count=int(count_text),
            even_dimension=int(even_text),
            decomposition=decomposition,
            nonisolated=any(marker in decomposition for marker in NONISOLATED_MARKERS),
        )
        if row.even_dimension != 2 + 2 * picard_rank:
            raise AssertionError(f"{model_id}: incorrect even-cohomology dimension")
        rows.append(row)
    if len(rows) != 105 or set(H12) != {row.identifier for row in rows}:
        raise AssertionError("the Hodge table and decomposition table do not contain the same 105 families")
    return rows


def summarize(rows: list[Row], details: bool) -> None:
    hodge_zero = [row for row in rows if row.h12 == 0]
    hodge_positive = [row for row in rows if row.h12 > 0]
    zero_nonisolated = [row for row in hodge_zero if row.nonisolated]
    zero_isolated = [row for row in hodge_zero if not row.nonisolated]
    zero_complete = [row for row in zero_isolated if row.milnor_total == row.euler_number]
    zero_deficient = [row for row in zero_isolated if row.milnor_total < row.euler_number]
    positive_nonisolated = [row for row in hodge_positive if row.nonisolated]
    positive_isolated = [row for row in hodge_positive if not row.nonisolated]
    positive_nonreduced = [
        row for row in hodge_positive if row.identifier in NONREDUCED_H12_POSITIVE
    ]
    positive_reduced_nonisolated = [
        row for row in positive_nonisolated
        if row.identifier not in NONREDUCED_H12_POSITIVE
    ]
    positive_reduced_isolated = [
        row for row in positive_isolated
        if row.identifier not in NONREDUCED_H12_POSITIVE
    ]
    problematic = zero_nonisolated + zero_deficient
    known_generic = [row for row in problematic if row.identifier in KNOWN_SMALL_QH_SEMISIMPLE]

    expected = (59, 46, 5, 54, 30, 24, 11, 15, 20, 21)
    actual = (
        len(hodge_zero), len(hodge_positive), len(zero_nonisolated),
        len(zero_isolated), len(zero_complete), len(zero_deficient),
        len(positive_nonreduced), len(positive_reduced_nonisolated),
        len(positive_reduced_isolated), len(known_generic),
    )
    if actual != expected:
        raise AssertionError(f"summary changed: {actual} != {expected}")
    if len(KNOWN_SMALL_QH_SEMISIMPLE) != 42:
        raise AssertionError("the generic-semisimplicity comparison should contain 42 families")

    print("PASS Hodge/Milnor audit for all 105 displayed Fano-threefold models")
    print(f"h12=0: {len(hodge_zero)}; h12>0: {len(hodge_positive)}")
    print(
        "h12=0: "
        f"non-isolated={len(zero_nonisolated)}, isolated Morse={len(zero_isolated)}, "
        f"Milnor total=e(F)={len(zero_complete)}, Milnor total<e(F)={len(zero_deficient)}, "
        "isolated non-Morse=0"
    )
    print(
        "h12>0: "
        f"non-reduced singular fiber={len(positive_nonreduced)}, "
        f"reduced with non-isolated critical locus={len(positive_reduced_nonisolated)}, "
        f"reduced with isolated Morse critical locus={len(positive_reduced_isolated)}"
    )
    print(
        f"Among the {len(problematic)} non-isolated-or-deficient h12=0 specializations, "
        f"{len(known_generic)} belong to families with known generically semisimple small quantum cohomology."
    )

    if details:
        def show(title: str, selected: list[Row]) -> None:
            print(f"\n{title}")
            for row in selected:
                mu = "--" if row.milnor_total is None else str(row.milnor_total)
                print(
                    f"  {row.identifier:5}  h12={row.h12:2}  "
                    f"mu={mu:>2}  e={row.euler_number:>3}  {row.label}"
                )

        show("h12=0, non-isolated", zero_nonisolated)
        show("h12=0, isolated Morse with Milnor deficit", zero_deficient)
        show("h12>0, non-reduced singular fiber", positive_nonreduced)
        show("h12>0, reduced and non-isolated", positive_reduced_nonisolated)
        show("h12>0, reduced and isolated Morse", positive_reduced_isolated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="Print the model lists.")
    parser.add_argument("--source", type=Path, default=Path("main.tex"))
    args = parser.parse_args()
    summarize(parse_rows(args.source), args.details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())