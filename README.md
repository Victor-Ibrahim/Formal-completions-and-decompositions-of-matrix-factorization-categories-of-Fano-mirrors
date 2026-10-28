# Formal Completions and Decompositions of Matrix Factorization Categories of Fano Mirrors

This repository and file package (`computations.zip`) contain the complete LaTeX source code for the scientific paper, as well as all the mathematical and computational validation scripts that support the demonstrated results[cite: 2].

## 📌 Project Structure

The repository is organized into several main directories to separate the algebraic computations from the LaTeX manuscript[cite: 2]:

### 1. Mathematical Computations and Validations (`computations/`)
This directory houses the Python scripts responsible for generating and auditing matrix factorizations and contact equivalences[cite: 2]:
* **Model Audits:** `input_model_audit.py`, `morse_bott_audit.py`, and `hodge_milnor_audit.py`[cite: 2].
* **Fano and Mori-Mukai Validations:** `rank_one_fano_validation.py` and `mori_mukai_validation.py` perform exact verification of the models[cite: 2].
* **Generation of Certificates and Correspondences:** `build_morimukai_correspondence.py` and `build_morimukai_certificates.py`[cite: 2]. These files produce the LaTeX appendices and CSV data (e.g., `supplementary-morimukai-certificates.tex`, `supplementary-morimukai-correspondence.csv`) that support the paper[cite: 2].
* **Gluing and Factorization:** `rank_one_five_chart_gluing.py`[cite: 2]. ### 2. Advanced Specific Verifications (`verification_Bd_V4d2/`)
A directory dedicated to verifications of specific hypersurfaces, combining computational scripts in Python and the software Singular (`.sing`)[cite: 2]:
* **Python Scripts:** `d1.py`, `d2.py`, `d3.py`, `d5.py`, `hodge.py`, `periods.py`, `periods2.py`, `jumps3.py`, `jumplib.py`, `recur.py`, and `common.py`[cite: 2].
* **Singular Scripts:** `b1types.sing`, `b2.sing`, `b4.sing`, `scan2.sing`, `tr.sing`, `types.sing`, `v14.sing`, `v18.sing`, `v6scan.sing`, among others[cite: 2].
* **Automation:** The `run_all.sh` file allows for the execution of the entire suite of tests and simulations in this directory at once[cite: 2].

### 3. LaTeX Source Code for the Paper (`sections/` and `Victor_paper_corrected_source/`)
Contains the complete scientific manuscript, organized into segments:
* **`sections/`:** Organizes the individual sections of the paper to facilitate compilation. It includes files ranging from `01-introduction.tex`, `02-preliminaries.tex`, and `03-supported-matrix-factorizations.tex` to specific cases such as `05-del-pezzo-models.tex`, `08-fano-rank-one-examples.tex`, `10-fano-higher-rank-examples.tex`, and `12-fano-hodge-compactification.tex`[cite: 2]. * **`Victor_paper_corrected_source/`:** Source code following corrections, containing `main_alex.tex`, bibliographic files (`ref.bib`, `main.bbl`), and notes on changes in `sections_3_4_and_dependent_corrections.diff` and `README_FOR_VICTOR.md`[cite: 2].
* **Root:** Global files such as `Appendix.tex` and `Bd_V4d+2_compactified_gluing.tex`[cite: 2].
* **`paper-reviews/`:** Contains the `.tex` file with the peer review of the document (`review-2026-10-01-112615.tex`)[cite: 2].
* **`prism-uploads/`:** arXiv submission files and manifests (`ARXIV_SOURCE_MANIFEST.txt` and draft PDF files such as `1303.3288v3 (1).pdf`)[cite: 2].

## 🚀 How to Run Computational Validations

To reproduce the results, generate mathematical certificates, or validate the Fano and Mori-Mukai models, execute the following commands from the repository root:

**Mori-Mukai Construction and Validation:**
```bash
python computations/build_morimukai_certificates.py
python computations/rank_one_fano_validation.py --all
python computations/mori_mukai_validation.py --all
