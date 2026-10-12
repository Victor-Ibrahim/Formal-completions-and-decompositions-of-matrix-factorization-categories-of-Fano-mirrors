#!/usr/bin/env python3
"""
Build a supplementary GRDB/Iskovskikh--Mori--Mukai correspondence table.

Input:  Apendice.tex
Output: supplementary-morimukai-correspondence.tex
        supplementary-morimukai-correspondence.csv

The GRDB source URL is recorded explicitly in the table.  The polynomial column
is extracted from the first displayed Laurent potential in each Mori--Mukai
example of Apendice.tex, so the table remains synchronized with the detailed
companion appendix.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

SOURCE_TEX = Path("Apendice.tex")
OUT_TEX = Path("supplementary-morimukai-correspondence.tex")
OUT_CSV = Path("supplementary-morimukai-correspondence.csv")
GRDB_URL = "http://www.grdb.co.uk/misc/morimukai.php"
EXPECTED_ENTRY_COUNT = 88


@dataclass(frozen=True)
class MoriMukaiEntry:
    label_raw: str
    label_tex: str
    label_key: str
    family: str
    polynomial: str
    source: str


def sort_key(label_key: str) -> tuple[int, int]:
    _, rank, number = label_key.split("_")
    return int(rank), int(number)


def strip_comments(line: str) -> str:
    if "%" not in line:
        return line
    before, _sep, _after = line.partition("%")
    return before.rstrip()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_display_blocks(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{aligned\}.*?\\end\{aligned\}", " ", text, flags=re.S)
    return text


def extract_family(block: str, label_tex: str) -> str:
    display_starts = [
        pos for pos in [
            block.find("\\["),
            block.find("$$"),
            block.find("\\begin{align"),
            block.find("\\fitdisplay{"),
        ]
        if pos >= 0
    ]
    prefix = block[: min(display_starts)] if display_starts else block[:800]
    prefix = remove_display_blocks(prefix)
    prefix = collapse_spaces(prefix)
    prefix = prefix.replace("\\,", " ")
    if prefix.startswith("Consider "):
        prefix = prefix[len("Consider "):]
    if not prefix:
        return f"Mori--Mukai family ${label_tex}$"
    sentence_match = re.match(r"(.+?\.)\s", prefix + " ")
    if sentence_match:
        return sentence_match.group(1).strip()
    return prefix[:300]


def find_display_bounds(lines: List[str], target_index: int) -> tuple[int, int]:
    start = target_index
    for idx in range(target_index, -1, -1):
        line = lines[idx]
        if "\\[" in line or "$$" in line or "\\begin{align" in line:
            start = idx
            break
    end = target_index
    for idx in range(target_index, len(lines)):
        line = lines[idx]
        if "\\]" in line or "\\end{align" in line or "\\end{aligned}" in line or "$$" in line and idx > start:
            end = idx
            break
    return start, end


def clean_polynomial_display(lines: Iterable[str]) -> str:
    text = "\n".join(strip_comments(line).strip() for line in lines if strip_comments(line).strip())
    replacements = [
        ("$$\\begin{aligned}", ""),
        ("\\begin{aligned}", ""),
        ("\\end{aligned}$$", ""),
        ("\\end{aligned}", ""),
        ("\\begin{align*}", ""),
        ("\\end{align*}", ""),
        ("\\[", ""),
        ("\\]", ""),
        ("$$", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.strip()
    if text.startswith("\\fitdisplay{"):
        text = text[len("\\fitdisplay{") :]
        if text.endswith("}"):
            text = text[:-1]
    text = re.sub(r"\\tag\{[^}]+\}", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text.strip()


def extract_polynomial(block: str, label_raw: str) -> str:
    lines = block.splitlines()
    target = f"MM_{{{label_raw}}}"
    candidates = []
    for idx, line in enumerate(lines):
        if target in line and "W" in line:
            candidates.append(idx)
    if not candidates:
        return f"W_{{MM_{{{label_raw}}}}}\\quad\\text{{(polynomial not found automatically)}}"

    target_idx = candidates[0]
    start, end = find_display_bounds(lines, target_idx)
    polynomial = clean_polynomial_display(lines[start : end + 1])
    if not polynomial:
        polynomial = clean_polynomial_display([lines[target_idx]])
    return polynomial


def latex_escape_text(text: str) -> str:
    # Keep math fragments and LaTeX commands from the manuscript intact.  Escape only table-sensitive characters
    # that occur in ordinary prose and are not expected as LaTeX commands.
    text = text.replace("&", "\\&")
    return text


def polycell(polynomial: str) -> str:
    return "\\polycell{" + polynomial + "}"


def familycell(family: str) -> str:
    return "\\familycell{" + latex_escape_text(family) + "}"


def extract_entries(source_tex: Path = SOURCE_TEX) -> List[MoriMukaiEntry]:
    text = source_tex.read_text(encoding="utf-8")
    blocks = re.findall(
        r"\\begin\{example\}(?:\[[^\]]*\])?(.*?)\\end\{example\}",
        text,
        flags=re.S,
    )
    entries: dict[str, MoriMukaiEntry] = {}
    for block in blocks:
        labels = re.findall(
            r"(?:\\widetilde\s*)?W_\{MM_\{([0-9]+-[0-9]+)\}\}",
            block,
        )
        if not labels:
            continue
        label_raw = labels[0]
        label_key = "MM_" + label_raw.replace("-", "_")
        if label_key in entries:
            continue
        rank, number = label_raw.split("-")
        label_tex = f"MM_{{{rank}-{number}}}"
        polynomial = extract_polynomial(block, label_raw)
        family = extract_family(block, label_tex)
        entries[label_key] = MoriMukaiEntry(
            label_raw=label_raw,
            label_tex=label_tex,
            label_key=label_key,
            family=family,
            polynomial=polynomial,
            source=f"GRDB Mori--Mukai table ({GRDB_URL}); polynomial transcribed from Apendice.tex.",
        )
    return [entries[key] for key in sorted(entries, key=sort_key)]


def write_csv(entries: List[MoriMukaiEntry], out_csv: Path = OUT_CSV) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["imm_id", "polynomial_latex", "fano_family", "source"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "imm_id": entry.label_key,
                    "polynomial_latex": entry.polynomial,
                    "fano_family": entry.family,
                    "source": entry.source,
                }
            )


def write_tex(entries: List[MoriMukaiEntry], out_tex: Path = OUT_TEX) -> None:
    rows = []
    for entry in entries:
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                f"${entry.label_tex}$",
                polycell(entry.polynomial),
                familycell(entry.family),
                "\\sourcecell{GRDB Mori--Mukai table; manuscript polynomial}",
            )
        )
    body = "\n".join(rows)
    out_tex.write_text(
        rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.7in,landscape]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{array,longtable,booktabs}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black]{{hyperref}}
\setlength{{\LTpre}}{{0.5em}}
\setlength{{\LTpost}}{{0.5em}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.25}}
\newcommand{{\polycell}}[1]{{\begingroup\scriptsize\(\begin{{aligned}}[t]#1\end{{aligned}}\)\endgroup}}
\newcommand{{\familycell}}[1]{{\begingroup\scriptsize #1\endgroup}}
\newcommand{{\sourcecell}}[1]{{\begingroup\scriptsize #1\endgroup}}
\begin{{document}}
\section*{{Supplementary GRDB/Iskovskikh--Mori--Mukai Correspondence Table}}

This table records the Iskovskikh--Mori--Mukai label used in the manuscript, the Laurent polynomial displayed there, the Fano family description attached to the example, and the external GRDB source to be used for cross-checking. The GRDB source page is \url{{{GRDB_URL}}}.
The polynomial column is generated from \texttt{{Apendice.tex}}; the machine-readable companion file is \texttt{{supplementary-morimukai-correspondence.csv}}.

\begin{{longtable}}{{>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.54\textwidth}}>{{\raggedright\arraybackslash}}p{{0.23\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}}}
\caption{{GRDB/Iskovskikh--Mori--Mukai correspondence for Mori--Mukai Laurent polynomial examples.}}\\
\toprule
\textbf{{IMM ID}} & \textbf{{Laurent polynomial}} & \textbf{{Fano family / manuscript description}} & \textbf{{Source}}\\
\midrule
\endfirsthead
\toprule
\textbf{{IMM ID}} & \textbf{{Laurent polynomial}} & \textbf{{Fano family / manuscript description}} & \textbf{{Source}}\\
\midrule
\endhead
\midrule
\multicolumn{{4}}{{r}}{{Continued on next page}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}

\bibliographystyle{{amsplain}}
\bibliography{{ref}}
\end{{document}}
""",
        encoding="utf-8",
    )


def main() -> int:
    entries = extract_entries()
    if len(entries) != EXPECTED_ENTRY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ENTRY_COUNT} Mori--Mukai entries in {SOURCE_TEX}, "
            f"found {len(entries)}."
        )
    write_csv(entries)
    write_tex(entries)
    print(f"Wrote {OUT_TEX} with {len(entries)} rows.")
    print(f"Wrote {OUT_CSV} with {len(entries)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())