"""Patch chapter4.tex with the details of the week-38 exercise-1 solution
(2026-09-16): RMSProp unrolling with the correct lower limit j=1, the weight
sum and memory, Adam's bias correction derived for both moments with the
correction factors tabulated, the first steps of Adam/RMSProp/uncorrected Adam
written out, and the feature-scaling argument done with S = diag(1,..,c,..,1),
followed by the reasons for standardising anyway.  No numbered equation is
added, so the equation numbers quoted in the week-38 material are unchanged.
Run from doc/BookML/."""
import sys
s = open("chapter4.tex").read()
orig = s

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (a[:80], n)
    s = s.replace(a, b)

# ---------------------------------------------------------------- 4.10 RMSProp: unrolling
rep(r'''\paragraph{Why the change matters.}
Expanding Eq.~(\ref{eq:4-rmspropaccum}) shows what has happened,
\begin{equation}
  \bm{v}_t = (1-\rho)\sum_{j=0}^{t}\rho^{\,t-j}\bm{g}_j^{2},
  \label{eq:4-rmspropsum}
\end{equation}
which is a weighted average with weights summing to nearly one, rather than an
unbounded sum.  Gradients older than roughly $1/(1-\rho)$ steps -- ten steps
for $\rho=0.9$, a hundred for $\rho=0.99$ -- are effectively forgotten.  Two
consequences follow.''',
r'''\paragraph{Why the change matters.}
Expanding Eq.~(\ref{eq:4-rmspropaccum}) shows what has happened.  Write
$\bm{g}_t=\nabla C(\bm{\theta}_t)$ for the gradient at step $t=1,2,\dots$, start
from $\bm{v}_0=\bm{0}$ and substitute the recursion into itself,
\begin{align*}
  \bm{v}_1 &= (1-\rho)\,\bm{g}_1^{2},\\
  \bm{v}_2 &= \rho\,\bm{v}_1+(1-\rho)\,\bm{g}_2^{2}
            = (1-\rho)\bigl[\rho\,\bm{g}_1^{2}+\bm{g}_2^{2}\bigr],\\
  \bm{v}_3 &= \rho\,\bm{v}_2+(1-\rho)\,\bm{g}_3^{2}
            = (1-\rho)\bigl[\rho^{2}\bm{g}_1^{2}+\rho\,\bm{g}_2^{2}+\bm{g}_3^{2}\bigr].
\end{align*}
The pattern is proved by induction: if
$\bm{v}_{t-1}=(1-\rho)\sum_{j=1}^{t-1}\rho^{\,t-1-j}\bm{g}_j^{2}$, then
$\bm{v}_t=\rho\,\bm{v}_{t-1}+(1-\rho)\bm{g}_t^{2}
=(1-\rho)\sum_{j=1}^{t-1}\rho^{\,t-j}\bm{g}_j^{2}+(1-\rho)\rho^{0}\bm{g}_t^{2}$,
that is,
\begin{equation}
  \bm{v}_t = (1-\rho)\sum_{j=1}^{t}\rho^{\,t-j}\bm{g}_j^{2}.
  \label{eq:4-rmspropsum}
\end{equation}
Gradient $\bm{g}_j$ therefore enters $\bm{v}_t$ with the weight
$w_{t,j}=(1-\rho)\rho^{\,t-j}$, and the weights form a finite geometric series,
\[
  \sum_{j=1}^{t}w_{t,j}=(1-\rho)\sum_{k=0}^{t-1}\rho^{k}
  =(1-\rho)\,\frac{1-\rho^{t}}{1-\rho}=1-\rho^{t}\;\longrightarrow\;1 .
\]
So $\bm{v}_t$ is a \emph{weighted mean} of the past squared gradients rather
than an unbounded sum; the missing mass $\rho^{t}$ is the weight still
carried by the zero initial value $\bm{v}_0$, a point Section~\ref{sec:adam}
returns to.  The mean is also a short one.  A gradient that entered at step
$j$ with weight $1-\rho$ has, $k=t-j$ steps later, the weight $(1-\rho)\rho^{k}$,
a fraction $\rho^{k}$ of its initial weight, which has fallen below $1/e$ once
\[
  \rho^{k}<e^{-1}\iff k>\frac{-1}{\ln\rho}\approx\frac{1}{1-\rho},
\]
using $-\ln\rho=(1-\rho)+\tfrac12(1-\rho)^{2}+\dots$ for $\rho$ close to one.
For $\rho=0.9$ this gives $-1/\ln\rho=9.5$, so the tenth step is the first at
which the weight has dropped below $1/e$ ($0.9^{9}=0.387$, $0.9^{10}=0.349$);
for $\rho=0.99$ it gives $99.5$ and the hundredth step ($0.99^{100}=0.366$).
Gradients older than roughly $1/(1-\rho)$ steps -- ten steps for $\rho=0.9$,
a hundred for $\rho=0.99$ -- are effectively forgotten, whereas in AdaGrad's
sum~(\ref{eq:4-adagradaccum}) every weight is one and the weights add up to
$t$.  Two consequences follow.''')

# ---------------------------------------------------------------- 4.11 Adam: bias correction
rep(r'''\paragraph{Bias correction.}
Those initialisations cause a problem which is worth deriving, because it
explains a step that otherwise looks arbitrary.  Suppose the gradient is
stationary with true second moment $\mathbb{E}[g^{2}]$.  Unrolling
Eq.~(\ref{eq:4-adamsecond}) as in Eq.~(\ref{eq:4-rmspropsum}) and taking the
expectation,
\begin{equation}
  \mathbb{E}[v_t] = (1-\beta_2)\sum_{j=1}^{t}\beta_2^{\,t-j}\,\mathbb{E}[g^{2}]
   = \mathbb{E}[g^{2}]\left(1-\beta_2^{\,t}\right),
  \label{eq:4-adambiasderiv}
\end{equation}
using the finite geometric sum.  The estimate is therefore too small by
exactly the factor $1-\beta_2^{t}$, and the same argument applies to
$\bm{m}_t$ with $\beta_1$.  The bias is severe at the start: with
$\beta_2=0.999$ the factor is $10^{-3}$ at $t=1$, so the raw $v_1$ underestimates
the true second moment by three orders of magnitude, and the step
$\gamma/\sqrt{v}$ would be enormous.  Dividing by the known factor removes the
bias exactly,
\begin{equation}
  \hat{\bm{m}}_t = \frac{\bm{m}_t}{1-\beta_1^{\,t}},
  \qquad
  \hat{\bm{v}}_t = \frac{\bm{v}_t}{1-\beta_2^{\,t}} .
  \label{eq:4-adambias}
\end{equation}
For small $t$ the correction is large, compensating for the initial zero; as
$t$ grows, $1-\beta_i^{t}\to1$ and the corrected moments converge to the raw
ones.  Bias correction is what makes Adam stable in its first iterations, and
it is the one ingredient AdaGrad and RMSProp lack.''',
r'''\paragraph{Bias correction.}
Those initialisations cause a problem which is worth deriving, because it
explains a step that otherwise looks arbitrary.  Equation~(\ref{eq:4-adamsecond})
has the form of Eq.~(\ref{eq:4-rmspropaccum}) with $\rho\to\beta_2$, so the
unrolled form~(\ref{eq:4-rmspropsum}) applies at once,
\[
  \bm{v}_t=(1-\beta_2)\sum_{j=1}^{t}\beta_2^{\,t-j}\bm{g}_j^{2},
  \qquad
  \bm{m}_t=(1-\beta_1)\sum_{j=1}^{t}\beta_1^{\,t-j}\bm{g}_j ,
\]
the second line being the same unrolling of Eq.~(\ref{eq:4-adamfirst}).
Suppose now that the gradient is stationary, with a true second moment
$\mathbb{E}[g_j^{2}]=\mathbb{E}[g^{2}]$ and a true mean
$\mathbb{E}[g_j]=\mathbb{E}[g]$ that do not depend on $j$ (the coordinate
index is suppressed).  The expectation is linear, so it passes through the
sum and leaves a finite geometric series,
\begin{equation}
  \mathbb{E}[v_t] = (1-\beta_2)\sum_{j=1}^{t}\beta_2^{\,t-j}\,\mathbb{E}[g^{2}]
   = \mathbb{E}[g^{2}]\,(1-\beta_2)\sum_{k=0}^{t-1}\beta_2^{\,k}
   = \mathbb{E}[g^{2}]\left(1-\beta_2^{\,t}\right),
  \label{eq:4-adambiasderiv}
\end{equation}
and identically $\mathbb{E}[m_t]=\mathbb{E}[g]\,(1-\beta_1^{\,t})$.  Each raw
moment is therefore too small by exactly the factor $1-\beta_i^{t}$, which is
the weight the exponential average still assigns to the zero initial value.
The bias is severe at the start: with $\beta_2=0.999$ the factor is $10^{-3}$
at $t=1$, so the raw $v_1$ underestimates the true second moment by three
orders of magnitude, and the step $\gamma/\sqrt{v}$ would be enormous.

The factors $1-\beta_1^{t}$ and $1-\beta_2^{t}$ are \emph{deterministic}: they
depend on $t$ and on the two constants, not on the data.  Dividing by a
constant commutes with the expectation, so
\begin{equation}
  \hat{\bm{m}}_t = \frac{\bm{m}_t}{1-\beta_1^{\,t}},
  \qquad
  \hat{\bm{v}}_t = \frac{\bm{v}_t}{1-\beta_2^{\,t}}
  \label{eq:4-adambias}
\end{equation}
satisfy $\mathbb{E}[\hat m_t]=\mathbb{E}[g]$ and
$\mathbb{E}[\hat v_t]=\mathbb{E}[g^{2}]$ for \emph{every} $t\ge1$, not only
asymptotically: the correction removes the bias exactly, and it does so by
rescaling rather than by averaging, so it adds no noise.  For small $t$ the
correction is large, compensating for the initial zero; as $t$ grows,
$1-\beta_i^{t}\to1$ and the corrected moments converge to the raw ones.  The
following values for the default constants show how different the two time
scales are:
\begin{center}
\begin{tabular}{lccc}
\hline
correction factor $1/(1-\beta_i^{t})$ & $t=1$ & $t=10$ & $t=1000$ \\
\hline
$\beta_1=0.9$   & $10$   & $1.54$ & $1.00$ \\
$\beta_2=0.999$ & $1000$ & $100$  & $1.58$ \\
\hline
\end{tabular}
\end{center}
The first moment is essentially unbiased after a few tens of steps, the second
needs of the order of $1/(1-\beta_2)=1000$ steps -- the memory of
Section~\ref{sec:rmsprop} with $\rho\to\beta_2$.  Stationarity is the
assumption that makes the correction exact; in a real run the gradient
changes as the iterate moves, and a residual bias remains, small because the
weights on old gradients are small.  Bias correction is what makes Adam
stable in its first iterations, and it is the one ingredient AdaGrad and
RMSProp lack.''')

# ---------------------------------------------------------------- 4.11 Adam: the first step
rep(r'''\paragraph{The size of an Adam step.}
Two properties of the update~(\ref{eq:4-adam}) follow from the algebra and
explain much of the method's behaviour in practice.  First, at $t=1$ the
bias-corrected moments are $\hat{\bm{m}}_1=\bm{g}_1$ and
$\hat{\bm{v}}_1=\bm{g}_1^{2}$, so the first step is
$\Delta\bm{\theta}_1=-\alpha\,\mathrm{sign}(\bm{g}_1)$ (up to $\epsilon$): a
move of length $\alpha$ in every coordinate, whatever the gradient, as for
AdaGrad.  Without the correction the first step would be
$\alpha(1-\beta_1)/\sqrt{1-\beta_2}\approx3.2\alpha$ with the default
constants -- the raw first moment is too small by a factor $10$, the raw
second moment by $10^{3}$, and the ratio inherits the mismatch.  (RMSProp,
which has no correction, starts with $v_1=(1-\rho)g_1^{2}$ and a step of
length $\gamma/\sqrt{1-\rho}$, ten times $\gamma$ for $\rho=0.99$: one more
reason for the correction.)  Second, at''',
r'''\paragraph{The size of an Adam step.}
Two properties of the update~(\ref{eq:4-adam}) follow from the algebra and
explain much of the method's behaviour in practice.  First, at $t=1$ the
raw moments are $\bm{m}_1=(1-\beta_1)\bm{g}_1$ and
$\bm{v}_1=(1-\beta_2)\bm{g}_1^{2}$, and the correction~(\ref{eq:4-adambias})
divides by exactly these factors, so $\hat{\bm{m}}_1=\bm{g}_1$ and
$\hat{\bm{v}}_1=\bm{g}_1^{2}$.  The first step is then
\[
  \Delta\theta_{1,j}=-\alpha\,\frac{g_{1,j}}{\sqrt{g_{1,j}^{2}}+\epsilon}
  =-\alpha\,\frac{g_{1,j}}{|g_{1,j}|+\epsilon}
  =-\alpha\,\mathrm{sign}(g_{1,j})\qquad(|g_{1,j}|\gg\epsilon),
\]
a move of length $\alpha$ in every coordinate, whatever the gradient -- a
gradient of $10^{-3}$ and one of $10^{3}$ produce the same first step, as for
AdaGrad in Eq.~(\ref{eq:4-adagradschedule}).  Without the correction the raw
moments would enter the update directly,
\[
  \Delta\theta_{1,j}=-\alpha\,\frac{(1-\beta_1)\,g_{1,j}}{\sqrt{1-\beta_2}\,|g_{1,j}|}
  =-\frac{1-\beta_1}{\sqrt{1-\beta_2}}\,\alpha\,\mathrm{sign}(g_{1,j})
  =-\frac{0.1}{0.0316}\,\alpha\,\mathrm{sign}(g_{1,j})\approx-3.2\,\alpha\,\mathrm{sign}(g_{1,j})
\]
with the default constants: the raw first moment is too small by a factor
$10$, the raw second moment by $10^{3}$, and the ratio inherits
$\sqrt{10^{3}}/10=3.16$.  RMSProp, which has no correction, starts with
$v_1=(1-\rho)g_1^{2}$ and Eq.~(\ref{eq:4-rmsprop}) gives
\[
  \Delta\theta_{1,j}=-\gamma\,\frac{g_{1,j}}{\sqrt{(1-\rho)g_{1,j}^{2}+\epsilon}}
  \approx-\frac{\gamma}{\sqrt{1-\rho}}\,\mathrm{sign}(g_{1,j}),
\]
a step of length $10\gamma$ for $\rho=0.99$ and $3.2\gamma$ for $\rho=0.9$,
which only settles to $\gamma\,\mathrm{sign}(g)$ after the $1/(1-\rho)$ steps
it takes the weights of Eq.~(\ref{eq:4-rmspropsum}) to sum to about one: one
more reason for the correction.  Second, at''')

# ---------------------------------------------------------------- 4.8 scale invariance
rep(r'''\paragraph{Scale invariance.}
A second consequence of Eq.~(\ref{eq:4-precond}) explains why the adaptive
methods are so much less sensitive to preprocessing than gradient descent.
Multiply feature $j$ by a constant $c$.  The parameter $\theta_j$ that
multiplies it is scaled by $1/c$ at the optimum, and by the chain rule the
gradient component $g_j$ is scaled by $c$; the accumulated second moment
$v_j$ is scaled by $c^{2}$, its square root by $c$, and the ratio
$g_j/\sqrt{v_j}$ in the update is unchanged.  The step $\Delta\theta_j$ is
therefore the same before and after -- in the rescaled units, that is, so the
iterate follows the same path in \emph{feature-normalised} coordinates
whatever the units of the data.  Plain gradient descent has no such property:
its step in coordinate $j$ scales by $c$ while the distance to the optimum
scales by $1/c$, and the stability bound~(\ref{eq:4-stability}) moves with
$c^{2}$.  Note what the invariance does not cover: it is an invariance to
the scaling of individual features, not to their correlation, which is
exactly the off-diagonal structure the previous paragraph showed to be out
of reach.''',
r'''\paragraph{Scale invariance.}
A second consequence of Eq.~(\ref{eq:4-precond}) explains why the adaptive
methods are so much less sensitive to preprocessing than gradient descent.
Multiply feature $j$, one column of the design matrix, by a constant $c$:
$\bm{X}'=\bm{X}\bm{S}$ with $\bm{S}=\mathrm{diag}(1,\dots,c,\dots,1)$.  The
least-squares cost~(\ref{eq:4-gdcost}) in the new parameters is
\[
  C'(\bm{\theta}')=\frac1n\|\bm{X}\bm{S}\bm{\theta}'-\bm{y}\|_2^{2}=C(\bm{S}\bm{\theta}'),
\]
so the same predictions are obtained with $\theta'_j=\theta_j/c$ and the
optimum moves to $\theta'^{*}_j=\theta^{*}_j/c$.  By the chain rule, or
directly from Eq.~(\ref{eq:4-gdgradient}),
\[
  \nabla C'(\bm{\theta}')=\bm{S}\,\nabla C(\bm{S}\bm{\theta}')
  =\frac2n\,\bm{S}\bm{X}^{T}(\bm{X}\bm{S}\bm{\theta}'-\bm{y}),
  \qquad\text{so}\qquad g'_j=c\,g_j,\quad g'_k=g_k\ (k\neq j),
\]
comparing the two problems at corresponding points $\bm{\theta}=\bm{S}\bm{\theta}'$,
and the Hessian~(\ref{eq:4-gdhessian}) becomes $\bm{H}'=\bm{S}\bm{H}\bm{S}$,
with $H'_{jj}=c^{2}H_{jj}$ and $H'_{jk}=c\,H_{jk}$ for $k\neq j$.  For plain
gradient descent the step in coordinate $j$ is $\Delta\theta'_j=-\gamma c\,g_j$:
it grows by $c$ while the distance to the optimum shrinks by $c$, a mismatch
of $c^{2}$, and this is what shifts the stability bound~(\ref{eq:4-stability})
through $\lambda_{\max}(\bm{H}')$, which for large $c$ grows like $c^{2}H_{jj}$.
A learning rate that was optimal before the rescaling diverges after it.
For the adaptive methods every accumulator is built from the gradient and its
square, so $r'_j=c^{2}r_j$, $v'_j=c^{2}v_j$ and $m'_j=c\,m_j$, and the ratios
in the updates are unchanged,
\[
  \frac{g'_j}{\sqrt{v'_j}}=\frac{c\,g_j}{c\sqrt{v_j}}=\frac{g_j}{\sqrt{v_j}},
  \qquad
  \frac{\hat m'_j}{\sqrt{\hat v'_j}}=\frac{\hat m_j}{\sqrt{\hat v_j}}.
\]
The step $\Delta\theta'_j=\Delta\theta_j$ does not depend on $c$ at all, as
long as $g_j^{2}\gg\epsilon$, and neither does the stability of the method:
the iterate takes the same steps in the rescaled units whatever the units of
the data.

\paragraph{Why one standardises nonetheless.}
The invariance just derived does not make the standardisation of
Chapter~\ref{chap:lreg} superfluous for the adaptive methods, for three
reasons.  First, it is an invariance to the scaling of individual features,
not to their correlation: the off-diagonal entries $H'_{jk}=cH_{jk}$ are
untouched by a diagonal $\bm{D}_t$, which is exactly the structure the
rotated bowl of Figure~\ref{fig:adaptivepaths} showed to be out of reach.
Centring the columns removes the largest single source of that correlation,
the one with the intercept, and leaves $\bm{H}$ as diagonal as the features
allow.  Second, the learning rate of an adaptive method is a length in the
units of each parameter (Section~\ref{sec:adam} makes this precise), and a
single $\gamma$ suits every coordinate only if the parameters are of
comparable size.  Scaling a column by $c$ leaves the step unchanged but
makes the distance $\theta^{*}_j/c$ it has to cover $c$ times shorter; with
$c=100$ a coordinate whose optimum is $0.06$ is walked at steps of
$\gamma=0.1$, straight past it.  Standardised columns give parameters of the
same order and let one $\gamma$ serve all of them.  Third, neither $\epsilon$
nor a penalty is invariant.  A feature scaled down until $g_j^{2}\lesssim\epsilon$
has its denominator dominated by $\epsilon$, and the update degenerates to
$-\gamma g_j/\sqrt{\epsilon}$, plain gradient descent with an enormous
learning rate; and the Ridge cost~(\ref{eq:4-ridgecost}) penalises
$\theta'^{2}_j=\theta_j^{2}/c^{2}$ instead of $\theta_j^{2}$, so the penalty,
and with it the minimiser, changes with the units -- the reason
Chapter~\ref{chap:lreg} standardises before regularising.''')

assert s != orig
open("chapter4.tex", "w").write(s)
print("chapter4.tex patched:", len(orig.splitlines()), "->", len(s.splitlines()), "lines")
