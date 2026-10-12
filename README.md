# Formal Completions and Decompositions of Matrix Factorization Categories of Fano Mirrors

Este repositório e pacote de ficheiros (`computations.zip`) contêm o código-fonte completo em LaTeX do artigo científico, bem como todos os scripts de validação matemática e computacional que dão suporte aos resultados demonstrados[cite: 2].

## 📌 Estrutura do Projeto

O repositório está organizado em múltiplas diretorias principais para separar as computações algébricas do manuscrito em LaTeX[cite: 2]:

### 1. Computações e Validações Matemáticas (`computations/`)
Esta diretoria aloja os scripts em Python responsáveis por gerar e auditar as fatorações de matrizes e as equivalências de contacto[cite: 2]:
* **Auditorias de Modelos:** `input_model_audit.py`, `morse_bott_audit.py` e `hodge_milnor_audit.py`[cite: 2].
* **Validações Fano e Mori-Mukai:** `rank_one_fano_validation.py` e `mori_mukai_validation.py` efetuam a verificação exata dos modelos[cite: 2].
* **Geração de Certificados e Correspondências:** `build_morimukai_correspondence.py` e `build_morimukai_certificates.py`[cite: 2]. Estes ficheiros produzem os anexos em LaTeX e dados em CSV (`supplementary-morimukai-certificates.tex`, `supplementary-morimukai-correspondence.csv`, etc.) que suportam o artigo[cite: 2].
* **Colagem e Fatoração:** `rank_one_five_chart_gluing.py`[cite: 2].

### 2. Verificações Específicas Avançadas (`verification_Bd_V4d2/`)
Diretoria dedicada a verificações de hipersuperfícies específicas, misturando scripts computacionais em Python e no software Singular (`.sing`)[cite: 2]:
* **Scripts Python:** `d1.py`, `d2.py`, `d3.py`, `d5.py`, `hodge.py`, `periods.py`, `periods2.py`, `jumps3.py`, `jumplib.py`, `recur.py` e `common.py`[cite: 2].
* **Scripts Singular:** `b1types.sing`, `b2.sing`, `b4.sing`, `scan2.sing`, `tr.sing`, `types.sing`, `v14.sing`, `v18.sing`, `v6scan.sing`, entre outros[cite: 2].
* **Automação:** O ficheiro `run_all.sh` permite executar toda a bateria de testes e simulações desta pasta de uma só vez[cite: 2].

### 3. Código-Fonte do Artigo em LaTeX (`sections/` e `Victor_paper_corrected_source/`)
Contém o manuscrito científico completo, devidamente segmentado:
* **`sections/`:** Organiza as secções individuais do artigo para facilitar a compilação. Inclui ficheiros desde `01-introduction.tex`, `02-preliminaries.tex` e `03-supported-matrix-factorizations.tex` até aos casos específicos `05-del-pezzo-models.tex`, `08-fano-rank-one-examples.tex`, `10-fano-higher-rank-examples.tex` e `12-fano-hodge-compactification.tex`[cite: 2].
* **`Victor_paper_corrected_source/`:** Código-fonte após correções, contendo `main_alex.tex`, ficheiros bibliográficos (`ref.bib`, `main.bbl`) e notas de diferenças em `sections_3_4_and_dependent_corrections.diff` e `README_FOR_VICTOR.md`[cite: 2].
* **Raiz:** Ficheiros globais como `Appendix.tex` e `Bd_V4d+2_compactified_gluing.tex`[cite: 2].
* **`paper-reviews/`:** Contém o ficheiro `.tex` com a revisão por pares do documento (`review-2026-10-01-112615.tex`)[cite: 2].
* **`prism-uploads/`:** Arquivos e manifestos de submissão do arXiv (`ARXIV_SOURCE_MANIFEST.txt` e ficheiros PDF de rascunho como `1303.3288v3 (1).pdf`)[cite: 2].

## 🚀 Como Executar as Validações Computacionais

Para reproduzir os resultados, gerar os certificados matemáticos ou validar os modelos de Fano e Mori-Mukai, execute os seguintes comandos a partir da raiz do repositório:

**Construção e Validação Mori-Mukai:**
```bash
python computations/build_morimukai_certificates.py
python computations/rank_one_fano_validation.py --all
python computations/mori_mukai_validation.py --all
