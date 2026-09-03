"""Extract every Python listing from the book chapters into BookPrograms/chapterN/."""
import re, os, ast
SRC=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST=f"{SRC}/BookPrograms"

TITLES={1:"linear_algebra",2:"statistics",3:"linear_regression",4:"optimization",
        5:"logistic_regression",6:"support_vector_machines",7:"trees_and_ensembles",8:"neural_networks",9:"differential_equations",10:"convolutional_networks",11:"recurrent_networks",12:"autoencoders",13:"transformers",14:"boltzmann",15:"vae",16:"diffusion",17:"gan"}

def strip_comments(s):
    """Strip LaTeX %-comments, but never inside a Python listing.

    Python code legitimately contains a bare %% (a format specifier, a modulo,
    a f-string percentage), and stripping it truncates the line and makes the
    listing unparseable, at which point it is silently mistaken for captured
    output and dropped.  We therefore pass verbatim environments through
    untouched.
    """
    out=[]; in_code=False
    for line in s.split("\n"):
        if re.match(r"\s*\\begin\{(Python|C\+\+)\}", line):
            in_code=True; out.append(line); continue
        if re.match(r"\s*\\end\{(Python|C\+\+)\}", line):
            in_code=False; out.append(line); continue
        if in_code:
            out.append(line); continue
        res=""
        for i,ch in enumerate(line):
            if ch=="%" and (i==0 or line[i-1]!="\\"): break
            res+=ch
        out.append(res)
    return "\n".join(out)

def section_of(tex, pos):
    """Nearest preceding \\section or \\subsection title."""
    best=("",-1)
    for m in re.finditer(r"\\(?:sub)?section\*?\{(.*?)\}", tex[:pos], re.S):
        if m.start()>best[1]: best=(m.group(1),m.start())
    return re.sub(r"[^a-z0-9]+","_",best[0].lower().replace("$","")).strip("_")[:40] or "listing"

total=0
for ch in range(1,18):
    tex=strip_comments(open(f"{SRC}/chapter{ch}.tex").read())
    d=f"{DST}/chapter{ch:02d}_{TITLES[ch]}"
    os.makedirs(d, exist_ok=True)
    n=0; names={}
    for m in re.finditer(r"\\begin\{Python\}\{\}\n(.*?)\\end\{Python\}", tex, re.S):
        code=m.group(1).rstrip("\n")
        try: ast.parse(code)
        except SyntaxError: continue            # captured output, not source
        n+=1
        base=section_of(tex, m.start())
        names[base]=names.get(base,0)+1
        suffix="" if names[base]==1 else f"_{names[base]}"
        fn=f"{d}/{n:02d}_{base}{suffix}.py"
        with open(fn,"w") as f:
            f.write(f'"""Chapter {ch}: listing {n}, from the section on '
                    f'{base.replace("_"," ")}.\n\nExtracted from doc/BookML/chapter{ch}.tex.\n"""\n\n')
            f.write(code+"\n")
        total+=1
    # a README per chapter
    with open(f"{d}/README.md","w") as f:
        f.write(f"# Chapter {ch} programs\n\n{n} listings extracted from "
                f"`doc/BookML/chapter{ch}.tex`.\nEach file is numbered in the order "
                f"it appears in the chapter.\n")
    print(f"chapter{ch:02d}_{TITLES[ch]}: {n} programs")
print("total:", total)
