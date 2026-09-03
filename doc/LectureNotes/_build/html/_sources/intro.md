# Machine Learning and the Physical Sciences

## Applied Data Analysis and Machine Learning — FYS-STK3155/4155

Welcome. This Jupyter-book is the executable companion to the textbook
*Machine Learning and the Physical Sciences: From discriminative methods to
generative methods and reinforcement learning*. It covers the same material as
the printed book, chapter for chapter, with every program in a cell you can
edit and run.

**Start with the [Introduction](introduction.ipynb).** It says what machine
learning is taken to mean here, what the book aims to do, how the seventeen
chapters fit together, which routes through them make sense, and what ethical
obligations come with the material. Everything that used to be summarised on
this page — the learning outcomes, the list of topics, the choice of
programming language — is developed there properly.

This page keeps only the practical matters: what you need to know beforehand,
and what you need installed.

## How the material is organised

The book is in five parts. Part I builds the mathematical and statistical
basis, Part II the classical methods of statistical learning, Part III deep
learning, Part IV generative modelling, and Part V looks towards reinforcement
learning. A closing chapter draws the whole together and says where these
methods are being taken in the sciences.

Each chapter develops the theory with the algebra shown, checks every result
that can be checked numerically against a program printed in the text, and
closes with a summary and a graded set of exercises. Every number quoted in
the text is produced by one of those programs, and every figure is generated
by running it, so anything here can be reproduced on a laptop in minutes.

Alongside the chapters you will find the weekly lecture notebooks, the weekly
exercise sets and the project descriptions for the current semester.

## Prerequisites and background

Basic knowledge of programming and of mathematics, with an emphasis on linear
algebra. Knowledge of Python and/or C++ is strongly recommended, and some
experience with Jupyter notebooks helps. The required courses are the
equivalents of the University of Oslo mathematics courses MAT1100, MAT1110 and
MAT1120, together with at least one of the corresponding programming courses
INF1000/INF1110 or MAT-INF1100/MAT-INF1100L/BIOS1100/KJM-INF1100. Most
universities now offer a basic, often compulsory, programming course in
Python.

We also recommend refreshing your knowledge of statistics and probability
theory; Chapter 2 of this book is written to serve as that refresher.

Computational work plays a central role and you are expected to work through
the numerical examples and projects that illustrate the theory. We strongly
recommend forming small project groups of two or three participants where
possible.

## Required technologies

Course participants are expected to have their own laptop. We use
[Git](https://git-scm.com/) as version control software, and using a provider
such as [GitHub](https://github.com/) or [GitLab](https://about.gitlab.com/)
is strongly recommended. If Git is new to you, this
[introduction](https://www.youtube.com/watch?v=RGOj5yH7evk) is a good starting
point.

We make extensive use of Python and its libraries. You can also use compiled
languages such as C++, Rust, Julia or Fortran where speed matters — the book
gives C++ and Fortran versions of several algorithms — but the focus of the
lectures is on Python.

If you have Python installed and are comfortable installing packages, the
following will cover most of what is needed:

```bash
pip install numpy scipy matplotlib ipython scikit-learn sympy pandas pillow
```

On macOS we recommend installing [Homebrew](https://brew.sh/) after Xcode, and
then `brew install python3`. On Linux, `sudo apt-get install python3` or the
equivalent for your distribution.

### Python distributions

If you would rather not set up dependencies and paths yourself,
[Anaconda](https://docs.anaconda.com/) is an open-source distribution of
Python and R for large-scale data processing, predictive analytics and
scientific computing, with package versions managed by `conda`.
[Google Colab](https://colab.research.google.com/) is a free Jupyter notebook
environment that requires no setup at all and runs entirely in the cloud.

### Useful Python libraries

- [NumPy](https://www.numpy.org/) — large multi-dimensional arrays and
  matrices, with a large collection of mathematical functions to operate on
  them. Used in every chapter.
- [SciPy](https://www.scipy.org/) — algorithms for mathematics, science and
  engineering built on NumPy.
- [Matplotlib](https://matplotlib.org/) — 2D plotting, producing
  publication-quality figures. Every figure in this book is made with it.
- [pandas](https://pandas.pydata.org/) — data structures and data analysis
  tools for labelled tabular data.
- [scikit-learn](https://scikit-learn.org/stable/) — simple and efficient
  tools for machine learning and data mining; the reference implementation we
  compare our own code against in the classical chapters.
- [PyTorch](https://pytorch.org/) and
  [TensorFlow](https://www.tensorflow.org/) — the two deep-learning
  frameworks. Chapters 10, 11, 13, 14, 15 and 17 each write the same model in
  both, and check the results against the implementation from scratch.
- [JAX](https://jax.readthedocs.io/en/latest/index.html) — composable
  transformations of Python and NumPy programs: automatic differentiation,
  vectorisation, parallelisation and just-in-time compilation. It has largely
  replaced [Autograd](https://github.com/HIPS/autograd).
- [SymPy](https://www.sympy.org/en/index.html) — symbolic mathematics, useful
  for checking a derivation by hand.
- [Xarray](https://docs.xarray.dev/en/stable/) — labelled multi-dimensional
  arrays, convenient for scientific data sets.

## Using and citing this material

All source files, programs and figures are freely available. You are
encouraged to use the code, modify it, and include it in publications, thesis
work and your own lectures. If you find it useful, a reference is appreciated:
M. Hjorth-Jensen, *Machine Learning and the Physical Sciences*, University of
Oslo.
