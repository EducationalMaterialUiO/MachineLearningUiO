# BookPrograms

Every Python listing appearing in the chapters of `doc/BookML`, extracted
verbatim, one directory per chapter.

| Directory | Chapter |
|---|---|
| `chapter01_linear_algebra` | 1 — Linear Algebra for Machine Learning |
| `chapter02_statistics` | 2 — Elements of Probability Theory and Statistics |
| `chapter03_linear_regression` | 3 — Linear Regression |
| `chapter04_optimization` | 4 — Optimization |
| `chapter05_logistic_regression` | 5 — Logistic Regression |
| `chapter06_support_vector_machines` | 6 — Support Vector Machines |
| `chapter07_trees_and_ensembles` | 7 — Ensemble methods |

Files are numbered in the order in which the listings appear in the chapter,
and named after the section they come from.  Several listings build on
definitions given in earlier listings of the same chapter (a class defined in
one, used in the next), so they are intended to be read in order rather than
each run in isolation.

The same code appears as executable cells in the Jupyter notebooks under
`doc/LectureNotes/chapterN.ipynb`.

## Regenerating

Two helper scripts live in this directory:

- `extract_programs.py` — rebuilds this tree from the chapter `.tex` files.
- `tex_to_notebook.py` — regenerates `doc/LectureNotes/chapterN.ipynb` from the
  same sources.

Both read `doc/BookML/chapterN.tex` directly, so the LaTeX remains the single
source of truth for the code; edit the chapter, then re-run the scripts.
