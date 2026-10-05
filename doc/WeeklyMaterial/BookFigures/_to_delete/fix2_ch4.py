s=open("chapter4.tex").read()
def rep(a,b):
    global s
    assert s.count(a)==1,(a[:60],s.count(a)); s=s.replace(a,b)
rep("""move of length $\\alpha$ in every coordinate, whatever the gradient, as for
AdaGrad and RMSProp.  Without the correction the first step would be
$\\alpha(1-\\beta_1)/\\sqrt{1-\\beta_2}\\approx3.2\\alpha$ with the default
constants -- the raw first moment is too small by a factor $10$, the raw
second moment by $10^{3}$, and the ratio inherits the mismatch.""","""move of length $\\alpha$ in every coordinate, whatever the gradient, as for
AdaGrad.  Without the correction the first step would be
$\\alpha(1-\\beta_1)/\\sqrt{1-\\beta_2}\\approx3.2\\alpha$ with the default
constants -- the raw first moment is too small by a factor $10$, the raw
second moment by $10^{3}$, and the ratio inherits the mismatch.  (RMSProp,
which has no correction, starts with $v_1=(1-\\rho)g_1^{2}$ and a step of
length $\\gamma/\\sqrt{1-\\rho}$, ten times $\\gamma$ for $\\rho=0.99$: one more
reason for the correction.)""")
rep("""one observes: RMSProp at a constant $\\gamma$ reaches an excess cost of
$10^{-4}$ and never $10^{-6}$, at every $\\gamma$ between $10^{-3}$ and $1$,""","""one observes: RMSProp at a constant $\\gamma$ reaches an excess cost of
$10^{-4}$ at best and never $10^{-6}$, at every $\\gamma$ between $10^{-3}$ and $1$,""")
rep("of $\\rho$, and the excess cost floors at about $\\lambda\\gamma^{2}/2$.","of $\\rho$, and the excess cost floors at a value of order $\\lambda\\gamma^{2}$.")
open("chapter4.tex","w").write(s)
bib=open("references.bib").read()
add=""
if "bottou2008" not in bib: add+="""
@inproceedings{bottou2008,
  author    = {Bottou, L\\'eon and Bousquet, Olivier},
  title     = {The tradeoffs of large scale learning},
  booktitle = {Advances in Neural Information Processing Systems 20},
  pages     = {161--168},
  year      = {2008}
}
"""
if "robbins1951" not in bib: add+="""
@article{robbins1951,
  author  = {Robbins, Herbert and Monro, Sutton},
  title   = {A stochastic approximation method},
  journal = {The Annals of Mathematical Statistics},
  volume  = {22},
  pages   = {400--407},
  year    = {1951}
}
"""
if "reddi2018" not in bib: add+="""
@inproceedings{reddi2018,
  author    = {Reddi, Sashank J. and Kale, Satyen and Kumar, Sanjiv},
  title     = {On the convergence of {Adam} and beyond},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}
"""
open("references.bib","a").write(add); print("fixed; bib added:", bool(add))
