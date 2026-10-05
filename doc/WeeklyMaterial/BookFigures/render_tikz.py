"""Render the TikZ schematics of the chapters to PNG for the Jupyter-book.

The book itself typesets these diagrams directly from the chapter source; the
notebooks cannot, so each figure environment containing a tikzpicture is
extracted, compiled standalone and written to
BookFigures/<dir>/<label>.png, where <label> is the part of
\\label{fig:...} after the colon.  tex_to_notebook.py looks for exactly that name.

Usage:  python3 render_tikz.py [sources, default 11]

A source is a chapter number (11) or the stem of a named front- or back-matter
file (conclusions, introduction), matching tex_to_notebook.py's SOURCES.
"""
import re, os, subprocess, sys, shutil, tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..")

# A diagram may label an arrow with Eq.~\ref{...}.  Compiled standalone there
# is no .aux to resolve it against, so the number would come out as "??";
# borrow the numbering tex_to_notebook.py builds from the sources themselves.
sys.path.insert(0, os.path.join(SRC, "BookPrograms"))
from tex_to_notebook import resolve_ref
# every source that contains a tikzpicture inside a figure environment
DIRS = {4:  "chapter04_optimization",
        5:  "chapter05_logistic_regression",
        8:  "chapter08_neural_networks",
        10: "chapter10_convolutional_networks",
        11: "chapter11_recurrent_networks",
        12: "chapter12_autoencoders",
        "conclusions": "conclusions"}

HEAD = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage[usenames,dvipsnames,x11names]{xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,arrows.meta,chains,positioning,fit,decorations.pathreplacing}
\begin{document}
"""

# labels may contain hyphens, as in \label{fig:concl-map}
LABEL = r"\\label\{fig:([A-Za-z0-9_-]+)\}"


def render(key):
    stem = f"chapter{key}" if isinstance(key, int) else key
    tex = open(os.path.join(SRC, f"{stem}.tex")).read()
    out = os.path.join(BASE, DIRS[key])
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in re.findall(r"\\begin\{figure\}\[[^\]]*\](.*?)\\end\{figure\}", tex, re.S):
        if "tikzpicture" not in f:
            continue
        lm = re.search(LABEL, f)
        if not lm:
            continue
        name = lm.group(1)
        body = re.sub(r"\\begin\{adjustbox\}.*?\n", "", f)
        body = body.replace("\\end{adjustbox}\n", "").replace("\\centering\n", "")
        body = re.sub(r"\\caption\{.*?\}\s*" + LABEL, "", body, flags=re.S)
        body = re.sub(r"\\ref\{([^}]*)\}",
                      lambda m: resolve_ref(m.group(1), True), body)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, name + ".tex")
            open(src, "w").write(HEAD + body + "\n\\end{document}\n")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", src],
                           cwd=tmp, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            pdf = os.path.join(tmp, name + ".pdf")
            if not os.path.exists(pdf):
                print(f"  !! {name}: pdflatex produced no output"); continue
            subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile",
                            pdf, os.path.join(out, name)], check=True)
            shutil.copy(pdf, os.path.join(out, name + ".pdf"))
        print(f"  {DIRS[key]}/{name}.png")
        n += 1
    return n


if __name__ == "__main__":
    args = [int(a) if a.isdigit() else a for a in sys.argv[1:]] or [11]
    print(f"rendered {sum(render(a) for a in args)} TikZ figures")
