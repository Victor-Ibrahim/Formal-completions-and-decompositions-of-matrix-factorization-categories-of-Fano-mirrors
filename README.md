# Formal Completions and Decompositions of Matrix Factorization Categories of Fano Mirrors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 About the Project

This repository provides the documentation, Python source code, `.sing` (Singular) scripts, and auxiliary LaTeX files supporting the article and the detailed computations in the research paper **Formal completions and decompositions of matrix factorization categories of Fano mirrors**. The focus is on performing formal verifications of local calculations, classifications, and validation regarding the use of matrix factorization categories, as well as the detailed constructions and decompositions surrounding Homological Mirror Symmetry, Fano varieties, and Landau-Ginzburg models.

## 📁 Repository Structure

* `build_morimukai_certificates.py`
* `rank_one_fano_validation.py`
* `mori_mukai_validation.py`
* `input_model_audit.py`
* `rank_one_five_chart_gluing.py`
## ⚙️ Prerequisites and Execution

The Python scripts dynamically generate and verify the certificates and logarithmic Hessian metrics for all 105 deformation families (15 "Rank One" classification models and 83 "Mori-Mukai" models, among other classification exceptions listed in the study guidelines).

To execute the complete verification pipeline for the "Mori-Mukai" examples:

```bash
# Generate the Mori-Mukai validation certificates
python computations/build_morimukai_certificates.py

# Validate the "Rank One" documentation cases
python computations/rank_one_fano_validation.py --all

# Run the global data check
python computations/mori_mukai_validation.py --all




# Computational verification — "Compactified gluing for the B_d and V_{4d+2} LG models"

Requirements: `python3` with `sympy`, and `Singular` (`apt install singular`).
Run the full suite: `./run_all.sh > saida_completa.txt` (the reference output is stored in `saida_completa.txt`).

| Script | Checks |
|---|---|
| `b4.py`, `b4.sing` | B4: eqs. (1)–(14); critical points, logarithmic and affine Hessians in all five charts, Jacobian radicals (4)–(8), and local types at q, a_X, and p_∞ |
| `v18.py`, `v18.sing`, `v18b.py` | V18: eqs. (21)–(36); p_a, Hessians modulo 3a²−1, Sing V(B), radicals (25)–(29), node p0, type A2 at r_X, and blow-up d_± |
| `d2.py`, `b2.sing`, `tr.sing` | B2/V10: (41)–(54); factorization with 21 monomials, P_a, q_±, r_a, det Q = 1/16, and central supports |
| `d3.py`, `v14.sing` | B3/V14: (59)–(74); factorization (61), P_±, r, eq. (72), strict transforms (73)–(74), and boundary node |
| `d1.py`, `b1types.sing` | B1/V6: (79)–(93); nodes q_α, det 13824, V6 at (1,1,1), λ=108, and det 209952 |
| `d5.py`, `v22b.sing`, `v22c.sing`, `types.sing` | B5/V22: (86)–(103); p_α, det 2(3α²+4), det 2(14α²+39α+32), Res=22, and blow-up (101)–(103) |
| `hodge.py` | Independent computations of e(X), H³, the index, and h^{1,2} (Chern classes / Atiyah–Bott localization), yielding k = h^{1,2}+1 |
| `periods.py`, `periods2.py` | Laurent-polynomial period sequences versus Fano-side quantum periods (I-functions; Grassmannians via the abelian/nonabelian correspondence) |
| `recur.py` | Picard–Fuchs operator inferred from the period; its singularities are the critical values, including boundary nodes |
| `scan2.sing` (`scan2.out`) | Singular locus of the total space {F_λ=0} over the base locus, for all pencils in P³ |
| `jumps3.py`, `v6jumps.py` | Jumps in the Milnor numbers of fibers at singular base points across the pencil |
