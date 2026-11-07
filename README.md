# Formal Completions and Decompositions of Matrix Factorization Categories of Fano Mirrors

Computations and verification scripts for the paper *Formal completions and
decompositions of matrix factorization categories of Fano mirrors* by
Victor Ibrahim Santos El Adji.

The paper computes, for specified Laurent-polynomial mirrors of the del Pezzo
surfaces and of the 105 deformation families of smooth Fano threefolds, the
critical loci, critical values, Hessians, and global formal normal forms on the
open torus. It also computes the resulting decompositions of the idempotent-completed
categories of matrix factorizations. This repository contains:

- the companion appendix with the model-by-model calculations;
- the scripts that check them;
- the generated supplements cited in the paper.

The paper's own LaTeX source is not included.

All files are in the top-level folder.

## Documents

| File | Contents |
| --- | --- |
| `Appendix.tex` | Companion appendix *Detailed Computations for the Fano Threefold Models*: polynomial identities, saturated critical ideals, critical points, parametrized components, Jacobian minors, Hessians and formal coordinate changes for each model. |
| `Bd_V4d+2_compactified_gluing.tex` | Compactification comparison for the five rank-one pairs $(B_d, V_{4d+2})$, $d = 1,\dots,5$. |
| `supplementary-morimukai-certificates.tex` | Certificate supplement for the Mori–Mukai entries, generated from `mori_mukai_validation.py`. |
| `supplementary-morimukai-correspondence.tex`, `.csv` | Correspondence between the Iskovskikh–Mori–Mukai labels, the Laurent polynomials used in the paper, and the GRDB source. |

## Scripts on the open torus (Python + SymPy)

| Script | Checks |
| --- | --- |
| `mori_mukai_validation.py` | Exact symbolic checks for the Mori–Mukai models: saturated logarithmic critical ideals, critical points and values, Hessians. Options `--list`, `--case MM_2_10`, `--all`, `--verbose`. |
| `rank_one_fano_validation.py` | The same checks for the 17 Picard-rank-one models. Options `--list`, `--case V_12`, `--all`. |
| `morse_bott_audit.py` | Every Morse–Bott component in the paper: critical equations, smoothness, Hessian rank equal to the codimension, a unit Jacobian minor for the global normal coordinates, and the discriminant square class. |
| `input_model_audit.py` | Source-level invariants of the input models, and the count benchmarks against the paper's tables. |
| `hodge_milnor_audit.py` | The Hodge/Milnor census of the 105 models (reduced versus nonreduced fibers, isolated versus non-isolated critical loci, $\mu_{\mathrm{tor}}$ against $e(F)$). Options `--details` and `--source <file>`. |
| `rank_one_five_chart_gluing.py` | Compactification audit for the rank-one comparison, on the four projective charts of $\mathbf P^3$ and the dense torus. |
| `build_morimukai_certificates.py` | Generates `supplementary-morimukai-certificates.tex`. |
| `build_morimukai_correspondence.py` | Generates `supplementary-morimukai-correspondence.tex` and `.csv` from the appendix. |

## Scripts for the rank-one compactifications (Python + Singular)

These scripts check the compactified pencils of the pairs $(B_d, V_{4d+2})$. They cover:

- critical points on the torus and on the charts;
- local singularity types and Milnor numbers;
- central fibers;
- Hodge invariants;
- quantum periods and Picard–Fuchs singularities.

| Files | Use |
| --- | --- |
| `b4.py`, `v18.py`, `v18b.py`, `d1.py`, `d2.py`, `d3.py`, `d5.py` | Models for $d = 4, 1, 2, 3, 5$. |
| `b1types.sing`, `b2.sing`, `b4.sing`, `v14.sing`, `v18.sing`, `v22b.sing`, `v22c.sing`, `tr.sing`, `types.sing`, `scan2.sing`, `v6scan.sing` | Singular checks of local types, Jacobian radicals and central fibers. |
| `hodge.py` | Euler characteristic, $h^{1,2}$, degree and index of the ten threefolds. |
| `periods.py`, `periods2.py`, `recur.py` | Quantum periods on the Fano side against constant terms of the Laurent polynomials, and the Picard–Fuchs singularities. |
| `jumps3.py`, `v6jumps.py`, `jumplib.py`, `common.py` | Milnor-number jumps at base points; shared helpers. |
| `run_all.sh` | Runs all of the above. |
| `saida_completa.txt`, `scan2.out` | Recorded output of `run_all.sh` and of `scan2.sing`. |

## Requirements

- Python 3 with SymPy.
- Singular (for example `apt install singular`) for the `.sing` files and `run_all.sh`.

## How to run

From the repository folder:

```bash
python3 mori_mukai_validation.py --all
python3 rank_one_fano_validation.py --all
python3 morse_bott_audit.py
python3 rank_one_five_chart_gluing.py
bash run_all.sh
```

## Known limitations

- `input_model_audit.py`, `hodge_milnor_audit.py`, and
  `mori_mukai_validation.py --coverage` read the paper's source file `main.tex`,
  which is not in this repository. `hodge_milnor_audit.py` accepts another path
  with `--source`.
- `build_morimukai_certificates.py` still looks for
  `computations/mori_mukai_validation.py`. In this flat layout the file is
  `mori_mukai_validation.py`.
- `build_morimukai_correspondence.py` reads `Apendice.tex`. In this repository
  the appendix is `Appendix.tex`.
