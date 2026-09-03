import re, json, os, sys, ast, uuid

# doc/BookML, derived from this script's own location (BookPrograms/..) so
# that the script runs unchanged wherever the repository is checked out.
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAPTERS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

# Front- and back-matter chapters, which are named rather than numbered.  The
# second entry is the number used when numbering their equations, tables and
# figures: the introduction sits in the front matter and is unnumbered (None),
# the conclusions are Chapter 18 of the printed book.
EXTRA = [("introduction", None), ("conclusions", 18)]

# every source the generator knows about, as (key, number) pairs
SOURCES = [(c, c) for c in CHAPTERS] + EXTRA

# BookFigures subdirectory per source, used for TikZ schematics that have no
# \includegraphics and are rendered separately by BookFigures/render_tikz.py.
# A source with no diagrams of that kind maps to None.
DIRS = {1:"chapter01_linear_algebra", 2:"chapter02_statistics",
        3:"chapter03_linear_regression", 4:"chapter04_optimization",
        5:"chapter05_logistic_regression", 6:"chapter06_support_vector_machines",
        7:"chapter07_trees_and_ensembles", 8:"chapter08_neural_networks",
        9:"chapter09_differential_equations",
        10:"chapter10_convolutional_networks", 11:"chapter11_recurrent_networks", 12:"chapter12_autoencoders", 13:"chapter13_transformers", 14:"chapter14_boltzmann", 15:"chapter15_vae", 16:"chapter16_diffusion", 17:"chapter17_gan",
        "introduction": None, "conclusions": "conclusions"}

MATH_ENVS = ("equation","equation*","align","align*","eqnarray","eqnarray*")

PREAMBLE = r"""$$
\newcommand{\bm}[1]{\boldsymbol{#1}}
\newcommand{\Det}[1]{|\boldsymbol{#1}|}
\newcommand{\bigO}{\mathcal{O}}
\newcommand{\var}{\mathrm{Var}}
\newcommand{\cov}{\mathrm{Cov}}
\newcommand{\Prob}{\mathrm{Prob}}
\newcommand{\mean}[1]{\langle #1 \rangle}
$$"""

def strip_comments(s):
    """Strip LaTeX %-comments, but never inside a verbatim listing, where a
    bare % is legitimate Python (a format specifier, a modulo, an f-string)."""
    out=[]; in_code=False
    for line in s.split("\n"):
        if re.match(r"\s*\\begin\{(Python|C\+\+)\}", line):
            in_code=True; out.append(line); continue
        if re.match(r"\s*\\end\{(Python|C\+\+)\}", line):
            in_code=False; out.append(line); continue
        if in_code:
            out.append(line); continue
        i=0; res=""
        while i < len(line):
            if line[i]=="%" and (i==0 or line[i-1]!="\\"):
                break
            res+=line[i]; i+=1
        out.append(res)
    return "\n".join(out)

def read(key):
    """Read a source by key: an int for chapterN.tex, a string for a named
    front- or back-matter file such as introduction.tex."""
    stem = f"chapter{key}" if isinstance(key, int) else key
    return strip_comments(open(f"{SRC}/{stem}.tex").read())

def chapter_title(s, key):
    """The \\chapter title, allowing the starred form used by the front matter."""
    m = re.search(r"\\chapter\*?(?:\[[^\]]*\])?\{(.*?)\}\s*\n", s, re.S)
    return (m.group(1).strip() if m else str(key)), m

# ---------------------------------------------------------------- pass 1
def build_maps():
    eqnum, secname, chapname, tabnum, fignum, thmnum = {}, {}, {}, {}, {}, {}
    for key, num in SOURCES:
        s = read(key)
        title, _ = chapter_title(s, key)
        # every chap: label, not only the first -- Chapter 7 carries both
        # \label{chap:trees} and \label{chap:ensemble}, and both are referenced.
        for lm in re.finditer(r"\\label\{(chap:[^}]*)\}", s):
            chapname[lm.group(1)] = (num, title)
        # section / subsection labels -> title
        for sm in re.finditer(r"\\(sub)?section\*?\{(.*?)\}\s*\n\s*\\label\{([^}]*)\}", s, re.S):
            secname[sm.group(3)] = sm.group(2).strip()
        if num is None:
            continue          # unnumbered source: nothing to number
        # table numbering
        tn = 0
        for tm2 in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", s, re.S):
            tn += 1
            for l in re.findall(r"\\label\{(tab:[^}]*)\}", tm2.group(1)): tabnum[l] = f"{num}.{tn}"
        # figure numbering, in document order
        fn = 0
        for fm2 in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S):
            fn += 1
            for l in re.findall(r"\\label\{(fig:[^}]*)\}", fm2.group(1)): fignum[l] = f"{num}.{fn}"
        # theorem-like environments.  book.tex loads svmono with envcountsame
        # and envcountchap, so all six share a single counter per chapter.
        tk = 0
        for tm3 in re.finditer(r"\\begin\{(theorem|proposition|lemma|corollary|definition|example)\}"
                               r"(.*?)\\end\{\1\}", s, re.S):
            tk += 1
            for l in re.findall(r"\\label\{([^}]*)\}", tm3.group(2)): thmnum[l] = f"{num}.{tk}"
        # equation numbering, in document order
        n = 0
        for em in re.finditer(r"\\begin\{(equation|align|eqnarray)\}(.*?)\\end\{\1\}", s, re.S):
            body = em.group(2)
            labels = re.findall(r"\\label\{([^}]*)\}", body)
            if em.group(1) in ("align","eqnarray"):
                # one number per \\ separated row that carries a label
                rows = body.split(r"\\")
                for r in rows:
                    ls = re.findall(r"\\label\{([^}]*)\}", r)
                    if "\\nonumber" in r: continue
                    n += 1
                    for l in ls: eqnum[l] = f"{num}.{n}"
            else:
                n += 1
                for l in labels: eqnum[l] = f"{num}.{n}"
    return eqnum, secname, chapname, tabnum, fignum, thmnum

EQ, SEC, CHAP, TAB, FIG, THM = build_maps()

def resolve_ref(label, mathmode=False):
    """Return the bare reference text; the source already supplies
    'Eq.~(...)', 'Chapter~', 'Section~' and so on.  Inside math a section
    title is set plain, since markdown emphasis would be shown literally."""
    if label in EQ:   return EQ[label]
    if label in TAB:  return TAB[label]
    if label in FIG:  return FIG[label]
    if label in THM:  return THM[label]
    if label in CHAP:
        num, title = CHAP[label]
        # the introduction has no number; name it instead
        if num is not None: return str(num)
        return title if mathmode else f"*{title}*"
    if label in SEC:  return SEC[label] if mathmode else f"*{SEC[label]}*"
    return label

def resolve_refs_in_math(text):
    """Cross-references also occur inside display math, in \\text{...} labels."""
    return re.sub(r"\\ref\{([^}]*)\}",
                  lambda m: resolve_ref(m.group(1), True), text)

# ---------------------------------------------------------------- inline
def inline(s):
    _st = []
    def _stash(m):
        _st.append(m.group(0)); return f"\x01M{len(_st)-1}\x01"
    s = re.sub(r"\$\$.*?\$\$", _stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", _stash, s)
    s = re.sub(r"\\(?:Eq|Section|Chapter|Table|Figure)?~?\\?ref\{([^}]*)\}",
               lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: f"[{m.group(1)}]", s)
    s = re.sub(r"\\index\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\emph\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s, flags=re.S)
    s = re.sub(r"\\textit\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\texttt\{(.*?)\}", lambda m: "`"+m.group(1).replace("\\_","_").replace("\\","")+"`", s, flags=re.S)
    s = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"<\1>", s)
    # a footnote whose whole content is a URL becomes a link on the word it is
    # attached to, which reads far better on screen than a parenthetical URL
    s = re.sub(r"([\w.\-)\]]+)([,.;:]?)\s*\\footnote\{<([^>]*)>\}",
               lambda m: f"[{m.group(1)}]({m.group(3)}){m.group(2)}", s)
    s = re.sub(r"\\footnote\{(.*?)\}", r" (\1)", s, flags=re.S)
    s = s.replace("\\%","%").replace("\\&","&").replace("\\#","#")
    s = re.sub(r"(?<!`)``(?!`)", '"', s)
    s = s.replace("''", '"')
    # text-mode LaTeX that legitimately appears outside math
    for a,b in [(r"\\ldots","..."), (r"\\dots","..."), (r"\\cdots","..."),
                (r"\\times"," x "), (r"\\ ", " "), (r"\\,", " "), (r"\\;", " "),
                (r"\\quad","  "), (r"\\qquad","    "), (r"\\hspace\{[^}]*\}"," "),
                (r"\\vspace\{[^}]*\}",""), (r"\\smallskip",""), (r"\\medskip",""),
                (r"\\bigskip",""), (r"\\/",""), (r"\\@","")]:
        s = re.sub(a, b, s)
    s = re.sub(r"\\-", "", s)
    s = s.replace("~"," ")
    s = re.sub(r"\\\\(?![a-zA-Z])", "  \n", s)
    s = re.sub(r"\x01M(\d+)\x01",
               lambda m: resolve_refs_in_math(_st[int(m.group(1))]), s)
    return s

MULTICOL = re.compile(r"\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}\s*$", re.S)

def tabular_to_md(body):
    body = re.sub(r"\\(top|mid|bottom)rule","",body)
    body = re.sub(r"\\addlinespace(\[[^\]]*\])?","",body)
    body = re.sub(r"\\cmidrule(\([^)]*\))?(\[[^\]]*\])?(\{[^}]*\})?","",body)
    body = re.sub(r"\\hline","",body)
    raw=[r.strip() for r in body.split(r"\\") if r.strip()]
    # a row is either a \multicolumn banner spanning the table, or ordinary cells
    rows=[]
    for r in raw:
        m = MULTICOL.match(r)
        if m and "&" not in r:
            rows.append((int(m.group(1)), [m.group(2)]))
        else:
            rows.append((None, [c.strip() for c in r.split("&")]))
    ncols = max([len(c) for span,c in rows if span is None] or [1])
    out=[]
    for i,(span,cells) in enumerate(rows):
        # a markdown cell must be a single line, so fold the LaTeX line breaks
        cells = [re.sub(r"\s+", " ", inline(c)).strip() for c in cells]
        cells += [""]*(ncols-len(cells))            # pad banners and short rows
        out.append("| "+" | ".join(cells[:ncols])+" |")
        if i==0: out.append("|"+"|".join(["---"]*ncols)+"|")
    return "\n".join(out)

def fignumber(inner):
    """'Figure 17.3: ' for a figure environment whose label we numbered."""
    lm = re.search(r"\\label\{(fig:[^}]*)\}", inner)
    return f"Figure {FIG[lm.group(1)]}: " if lm and lm.group(1) in FIG else ""

def tabnumber(inner):
    lm = re.search(r"\\label\{(tab:[^}]*)\}", inner)
    return f"Table {TAB[lm.group(1)]}: " if lm and lm.group(1) in TAB else ""

def convert_prose(s, chnum):
    s = re.sub(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}", "", s)

    # figures -> markdown image + italic caption.  The .pdf used by LaTeX has a
    # .png twin written by the same BookFigures script; point at the .png and
    # make the path relative to doc/LectureNotes.
    def fig_repl(m):
        inner = m.group(1)
        gm = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", inner)
        cm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", inner, re.S)
        if not gm:
            # A TikZ schematic rather than a generated plot.  These are rendered
            # to PNG by BookFigures/render_tikz.py and named after their label.
            if "tikzpicture" in inner and DIRS.get(chnum):
                lm = re.search(r"\\label\{fig:([A-Za-z0-9_-]+)\}", inner)
                if lm:
                    cm2 = re.search(r"\\caption\{(.*?)\}\s*\\label", inner, re.S)
                    cap2 = inline(cm2.group(1)).strip().replace("\n", " ") if cm2 else ""
                    alt2 = re.sub(r"[^A-Za-z0-9 ,.-]", "",
                                  re.sub(r"\$[^$]*\$", "", cap2)).strip()[:70]
                    p = f"../BookML/BookFigures/{DIRS[chnum]}/{lm.group(1)}.png"
                    return f"\n\n![{alt2}]({p})\n\n*{fignumber(inner)}{cap2}*\n\n"
            return ""
        path = "../BookML/" + gm.group(1) + ".png"
        cap = inline(cm.group(1)).strip().replace("\n", " ") if cm else ""
        alt = re.sub(r"\\$[^$]*\\$", "", cap)          # no math in the alt text
        alt = re.sub(r"[^A-Za-z0-9 ,.-]", "", alt).strip()[:70]
        return f"\n\n![{alt}]({path})\n\n*{fignumber(inner)}{cap}*\n\n"
    s = re.sub(r"\\begin\{figure\}\[?[^\]]*\]?(.*?)\\end\{figure\}",
               fig_repl, s, flags=re.S)
    # tables
    def table_repl(m):
        inner=m.group(1)
        # the column specification may itself contain braces, as in {@{}clp{9cm}@{}}
        tm=re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}",
                     inner, re.S)
        cm=re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\Z)", inner, re.S)
        md = tabular_to_md(tm.group(1)) if tm else ""
        cap = ("\n\n*"+tabnumber(inner)+inline(cm.group(1)).strip()+"*") if cm else ""
        return "\n\n"+md+cap+"\n\n"
    s = re.sub(r"\\begin\{table\}\[?[^\]]*\]?(.*?)\\end\{table\}", table_repl, s, flags=re.S)
    s = re.sub(r"\\begin\{center\}(.*?)\\end\{center\}", lambda m: table_repl(m), s, flags=re.S)

    # notebox -> admonition
    def note_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        title = "Note"
        tm = re.match(r"\*\*(.*?)\*\*\s*", inner)
        if tm:
            title = tm.group(1).rstrip(".")
            inner = inner[tm.end():]
        body = "\n".join("  "+l if l.strip() else "" for l in inner.split("\n"))
        return f"\n\n```{{admonition}} {title}\n:class: tip\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{notebox\}(.*?)\\end\{notebox\}", note_repl, s, flags=re.S)

    # theorem-like environments -> titled admonitions (Chapter 10 onwards)
    THMENV = {"theorem": "Theorem", "proposition": "Proposition",
              "lemma": "Lemma", "corollary": "Corollary",
              "definition": "Definition", "example": "Example"}
    def thm_repl(env, cls):
        def _f(m):
            opt, body = m.group(1), m.group(2)
            name = re.sub(r"^\[|\]$", "", opt or "").strip()
            lab = re.search(r"\\label\{([^}]*)\}", body)
            numtxt = f" {THM[lab.group(1)]}" if lab and lab.group(1) in THM else ""
            title = THMENV[env] + numtxt + (f" ({name})" if name else "")
            inner = convert_prose(body, chnum).strip()
            return f"\n\n```{{admonition}} {title}\n:class: {cls}\n{inner}\n```\n\n"
        return _f
    for env, cls in [(e, "important") for e in THMENV]:
        s = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(.*?)\\end\{" + env + r"\}",
                   thm_repl(env, cls), s, flags=re.S)
    def proof_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        return f"\n\n```{{admonition}} Proof\n:class: note\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{proof\}(.*?)\\end\{proof\}", proof_repl, s, flags=re.S)

    # display math
    def math_repl(m):
        env, body = m.group(1), m.group(2)
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        tag = ""
        if labels and labels[0] in EQ: tag = "\\tag{%s}" % EQ[labels[0]]
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        if env.startswith("equation"):
            return "\n\n$$\n"+body.strip()+tag+"\n$$\n\n"
        return "\n\n$$\n\\begin{"+env+"}\n"+body.strip()+"\n\\end{"+env+"}\n$$\n\n"
    s = re.sub(r"\\begin\{(equation\*?|align\*?|eqnarray\*?)\}(.*?)\\end\{\1\}", math_repl, s, flags=re.S)
    s = re.sub(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]",
               lambda m: "\n\n$$\n"+m.group(1).strip()+"\n$$\n\n", s, flags=re.S)

    # lists, innermost first so that nesting is handled correctly
    def list_repl(body, ordered, indent):
        items = re.split(r"\\item\s", body)[1:]
        pad = "    " * indent
        out = []
        for i, t in enumerate(items):
            t = inline(t).strip()
            marker = f"{i+1}." if ordered else "-"
            lines = t.split("\n")
            first = pad + marker + " " + lines[0].strip()
            rest = [pad + "   " + l.strip() if l.strip() else "" for l in lines[1:]]
            out.append("\n".join([first] + rest))
        return "\n\n" + "\n".join(out) + "\n\n"

    def expand_lists(text, indent=0):
        pat = re.compile(r"\\begin\{(itemize|enumerate)\}"
                         r"((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)"
                         r"\\end\{\1\}", re.S)
        while True:
            m = pat.search(text)
            if not m: break
            text = text[:m.start()] + list_repl(m.group(2), m.group(1)=="enumerate",
                                                indent) + text[m.end():]
        return text
    s = expand_lists(s)

    # sectioning.  The title may itself contain braces -- a heading such as
    # \subsection{Against Chapter~\ref{chap:lreg}} is common -- so the argument
    # is matched with one level of nesting allowed rather than non-greedily.
    ARG = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
    s = re.sub(r"\\section\*?"+ARG, lambda m: "\n\n## "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\subsection\*?"+ARG, lambda m: "\n\n### "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\paragraph"+ARG, lambda m: "\n\n**"+inline(m.group(1)).rstrip(".")+".** ", s, flags=re.S)
    s = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\chaptermark\{[^}]*\}", "", s)
    # front-matter bookkeeping that carries no meaning in a notebook
    s = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\(clearemptydoublepage|newpage|noindent|centering|toprule|midrule|bottomrule|addlinespace|phantomsection)\b", "", s)
    s = re.sub(r"\\(small|footnotesize|scriptsize|normalsize|large|Large)\b", "", s)
    s = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", s)

    # protect math so that the text-mode substitutions in inline() cannot
    # reach inside it
    store = []
    def stash(m):
        store.append(m.group(0)); return f"\x00MATH{len(store)-1}\x00"
    s = re.sub(r"\$\$.*?\$\$", stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", stash, s)

    s = inline(s)

    s = re.sub(r"\x00MATH(\d+)\x00",
               lambda m: resolve_refs_in_math(store[int(m.group(1))]), s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def convert_chapter(key, num=None):
    """Convert one source.  `key` selects the file, `num` the chapter number
    used in the heading; a source with num=None gets an unnumbered heading."""
    if num is None and isinstance(key, int): num = key
    ch = key                                  # figure-directory key
    s = read(key)
    title, m = chapter_title(s, key)
    s = s[m.end():]
    s = re.sub(r"^\s*\\chaptermark\{[^}]*\}\s*\n","",s)
    s = re.sub(r"^\s*\\label\{[^}]*\}\s*\n","",s)
    head = f"# Chapter {num}: {title}" if num is not None else f"# {title}"

    cells=[{"cell_type":"markdown","metadata":{},
            "source":[head+"\n"]},
           {"cell_type":"markdown","metadata":{},
            "source":["<!-- Macro definitions for MathJax, mirroring book.tex -->\n", PREAMBLE]}]

    parts = re.split(r"\\begin\{Python\}\{\}\n(.*?)\\end\{Python\}", s, flags=re.S)
    for i,part in enumerate(parts):
        if i % 2 == 0:
            md = convert_prose(part, ch)
            if md:
                # split into one cell per section/subsection so the notebook
                # stays navigable rather than one enormous cell
                chunks, cur = [], []
                for line in md.split("\n"):
                    if line.startswith("## ") and cur and any(x.strip() for x in cur):
                        chunks.append("\n".join(cur).strip()); cur=[line]
                    else:
                        cur.append(line)
                if cur: chunks.append("\n".join(cur).strip())
                for chunk in chunks:
                    if chunk:
                        cells.append({"cell_type":"markdown","metadata":{},
                                      "source":[l+"\n" for l in chunk.split("\n")]})
        else:
            code = part.rstrip("\n")
            try:
                ast.parse(code); is_output = False
            except SyntaxError:
                is_output = True          # captured program output, not source
            if is_output:
                cells.append({"cell_type":"markdown","metadata":{},
                              "source":["```\n"]+[l+"\n" for l in code.split("\n")]+["```\n"]})
            else:
                cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                              "outputs":[], "source":[l+"\n" for l in code.split("\n")]})
    for c in cells: c["id"] = uuid.uuid4().hex[:12]
    nb={"cells":cells,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    return nb

if __name__ == "__main__":
    out = sys.argv[1]
    # optional second argument: a comma separated list of sources to rebuild,
    # e.g. "17,introduction,conclusions".  Default is everything.
    wanted = None
    if len(sys.argv) > 2:
        wanted = {int(x) if x.strip().isdigit() else x.strip()
                  for x in sys.argv[2].split(",")}
    os.makedirs(out, exist_ok=True)
    for key, num in SOURCES:
        if wanted is not None and key not in wanted: continue
        nb = convert_chapter(key, num)
        stem = f"chapter{key}" if isinstance(key, int) else key
        with open(f"{out}/{stem}.ipynb","w") as f: json.dump(nb,f,indent=1)
        nmd=sum(1 for c in nb["cells"] if c["cell_type"]=="markdown")
        ncode=sum(1 for c in nb["cells"] if c["cell_type"]=="code")
        print(f"{stem}.ipynb: {nmd} markdown, {ncode} code cells")
