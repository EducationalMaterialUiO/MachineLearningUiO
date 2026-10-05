"""Patch chapter4.tex with the week-38 material (2026-09-10).  Run from doc/BookML/."""
import re, sys
s = open("chapter4.tex").read()
orig = s

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (a[:80], n)
    s = s.replace(a, b)

# ---------------------------------------------------------------- 4.6 momentum: following the steps
rep(r'''For this one-dimensional problem $\kappa=1$ and there is nothing to fix, yet
momentum still helps by increasing the effective step through
Eq.~(\ref{eq:4-momentumgain}).  The dramatic gains appear when the eigenvalues
differ, which is the case for every realistic problem.
''', r'''For this one-dimensional problem $\kappa=1$ and there is nothing to fix, yet
momentum still helps by increasing the effective step through
Eq.~(\ref{eq:4-momentumgain}).  The dramatic gains appear when the eigenvalues
differ, which is the case for every realistic problem.

\paragraph{Following the steps.}
It is worth watching the two iterations step by step on a function simple
enough to be drawn.  The function \texttt{descend} below is the whole of
Eq.~(\ref{eq:4-momentum}), with the gradient passed in as an argument -- the
same structure we keep throughout the chapter, so that the analytical
gradient of Section~\ref{sec:gdlinreg} and the automatic one of
Section~\ref{sec:autodiff} are interchangeable.

\begin{Python}{}
import numpy as np

def quartic(x):
    """A low-order polynomial with two minima: f(x) = x^4 - 3x^2 + x."""
    return x**4 - 3.0 * x**2 + x

def quartic_grad(x):
    return 4.0 * x**3 - 6.0 * x + 1.0

def descend(grad, x0, gamma, beta=0.0, num_iters=40):
    """Gradient descent with momentum, Eq. (4.momentum); beta = 0 is plain gradient descent.
    Works for a scalar x as well as for a vector theta.  Returns every iterate."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(num_iters):
        v = beta * v + gamma * grad(x)
        x = x - v
        history.append(x.copy())
    return np.array(history)

for beta in (0.0, 0.7):
    path = descend(quartic_grad, 2.0, gamma=0.05, beta=beta)
    print(f"beta = {beta}: x = {np.round(path[:6], 3)} ... ends at {path[-1]:.4f}, f = {quartic(path[-1]):.4f}")
\end{Python}

The quartic $f(x)=x^{4}-3x^{2}+x$ has a global minimum at $x=-1.30$
($f=-3.51$), a local one at $x=1.13$ ($f=-1.07$), and a hump at $x=0.17$
between them.  Started at $x_0=2$ with $\gamma=0.05$, plain gradient descent
walks down into the \emph{nearest} minimum, $2\to0.95\to1.01\to1.06\to\dots\to1.131$,
and stays there, exactly as Section~\ref{sec:graddescent} warned.  With
$\beta=0.7$ the velocity built up on the steep slope carries the iterate over
the hump, $2\to0.95\to0.28\to-0.16\to-0.57\to\dots\to-1.302$, into the global
minimum: the same code, one extra line, a different answer.  This should not
be over-read.  Momentum knows nothing of the landscape beyond the gradients it
has seen; it crossed the hump because the accumulated velocity happened to
exceed the barrier, and with $\beta=0.9$ the ``ball'' is so frictionless that
it overshoots the global minimum, climbs back over the hump and ends in the
local basin.  Once the cost is not convex, none of the methods of this chapter
comes with a guarantee, and Section~\ref{sec:convexity} is the last time one
was available.

The right panel of Fig.~\ref{fig:gdsteps} repeats the exercise on the bowl
$C(\bm{\theta})=\frac12(\theta_1^{2}+10\,\theta_2^{2})$, whose Hessian is
$\mathrm{diag}(1,10)$, so that $\kappa=10$ and $2/\lambda_{\max}=0.2$.  From
$(2,1.4)$, forty steps at $\gamma=0.04$ creep along the valley floor (356
iterations to reach $10^{-6}$); forty steps at $\gamma=0.18$, just below the
bound, zigzag across the steep direction while advancing slowly along the
flat one (74 iterations); momentum with $\beta=0.3$ at the same $\gamma$
cancels the zigzag and arrives in 39, and the tuned pair
$\gamma=4/(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^{2}=0.231$,
$\beta=\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^{2}=0.27$ that attains the
rate~(\ref{eq:4-momentumrate}) needs 28.  Note that the best $\beta$ is tied
to $\kappa$: the default $0.9$ is right for $\kappa\approx400$ and too much
for $\kappa=10$, where it makes the iterate ring for longer than plain
descent takes to converge.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.98\textwidth]{BookFigures/chapter04_optimization/gd_steps}
\caption{Following the steps of gradient descent with and without momentum.  Left: the quartic $f(x)=x^{4}-3x^{2}+x$ from $x_0=2$ with $\gamma=0.05$; plain descent stops in the nearer, shallower minimum, momentum with $\beta=0.7$ carries the iterate over the hump into the global one.  Right: forty steps on the bowl $\frac12(\theta_1^{2}+10\theta_2^{2})$ with $\kappa=10$, at a small learning rate, at $0.9$ of the stability bound (\ref{eq:4-stability}), and with momentum $\beta=0.3$ at the same learning rate.}
\label{fig:gdsteps}
\end{figure}
''')

# ---------------------------------------------------------------- 4.7 SGD: estimator, flops, example
rep(r'''By Section~\ref{sec:clt}, the noise in a minibatch gradient estimate falls as
$1/\sqrt{M}$: quadrupling the batch size halves the gradient noise at four
times the cost per step.  That unfavourable exchange rate is why very large
batches are not automatically better.
''', r'''By Section~\ref{sec:clt}, the noise in a minibatch gradient estimate falls as
$1/\sqrt{M}$: quadrupling the batch size halves the gradient noise at four
times the cost per step.  That unfavourable exchange rate is why very large
batches are not automatically better.

\paragraph{The minibatch gradient as an estimator.}
Let us make the statement about noise precise.  Write the full gradient as
the average $\bm{g}=\frac1n\sum_i\bm{g}_i$ of the per-example gradients
$\bm{g}_i=\nabla c_i$ (for the cost $\frac1n\|\bm{X}\bm{\theta}-\bm{y}\|^{2}$
this is exactly Eq.~(\ref{eq:4-gdgradient}), and the minibatch version
$\frac{2}{M}\bm{X}_B^{T}(\bm{X}_B\bm{\theta}-\bm{y}_B)$ is the same formula
on the rows of the batch).  A minibatch drawn uniformly without replacement
gives the estimate $\hat{\bm{g}}_B=\frac1M\sum_{i\in B}\bm{g}_i$, and the
elementary theory of sampling from a finite population yields
\begin{equation}
  \mathbb{E}[\hat{\bm{g}}_B]=\bm{g},
  \qquad
  \mathrm{Var}[\hat g_{B,j}]=\frac{s_j^{2}}{M}\,\frac{n-M}{n-1},
  \qquad
  s_j^{2}=\frac{1}{n-1}\sum_{i=1}^{n}\bigl(g_{i,j}-g_j\bigr)^{2}.
  \label{eq:4-minibatchvar}
\end{equation}
The estimate is \emph{unbiased}, whatever $M$; its standard deviation falls
as $1/\sqrt{M}$, times the finite-population factor $\sqrt{(n-M)/(n-1)}$,
which is one for $M\ll n$ and vanishes at $M=n$, where the estimate is the
full gradient.  Drawing with replacement, as the \texttt{randint} loop above
does, drops the factor.  Two things follow.  The direction of an SGD step is
right on average and wrong in detail, by an amount that depends on
$\bm{\theta}$ through the spread $s_j$ of the per-example gradients -- large
far from the minimum, and not zero at the minimum, since there the
per-example gradients cancel in the sum without vanishing individually.  And
the noise does not disappear with more iterations, only with a larger $M$ or
a smaller step; the consequences of this are worked out in
Section~\ref{sec:schedules}.

\paragraph{Counting floating-point operations.}
The claim that SGD is ahead per unit of computation can be made exact for the
least-squares cost, whose gradient~(\ref{eq:4-gdgradient}) is two
matrix--vector products: $\bm{X}\bm{\theta}$ costs $2np$ floating-point
operations (one multiplication and one addition per matrix element),
subtracting $\bm{y}$ costs $n$, and $\bm{X}^{T}\bm{r}$ another $2np$, so
\begin{equation}
  \text{one full gradient}\approx4np\ \text{flops},
  \qquad
  \text{one minibatch gradient}\approx4Mp\ \text{flops},
  \label{eq:4-flops}
\end{equation}
and an epoch of SGD -- $n/M$ updates -- costs the same $4np$ as a single
step of gradient descent.  For comparison, solving the normal equations
directly costs about $np^{2}+p^{3}/3$ flops (forming
$\bm{X}^{T}\bm{X}$, then a Cholesky factorisation, Section~\ref{sec:lu}),
which is $p/4$ gradient evaluations: for a linear model with modest $p$ the
closed form is the cheapest option of all, and the comparison below is
between the two \emph{iterative} methods, which are what remains once the
model is a network and no closed form exists.

Now count what each method needs.  Gradient descent at its best learning
rate converges at the rate~(\ref{eq:4-gdrate}), so reaching an excess cost
$\epsilon$ takes about $\frac{\kappa}{2}\ln(C_0/\epsilon)$ iterations and
$2np\kappa\ln(C_0/\epsilon)$ flops; momentum improves $\kappa$ to
$\sqrt{\kappa}$.  Stochastic gradient descent with a decaying learning rate
reaches excess cost $\epsilon$ after a number of \emph{updates} proportional
to $1/\epsilon$ (the $\bigO(1/k)$ rate above), each costing $4Mp$; the count
does not involve $n$ at all.  The comparison therefore turns on what
$\epsilon$ is worth aiming for, and here statistics settles the matter.  The
minimiser $\hat{\bm{\theta}}$ of the empirical cost is itself only an
estimate of the parameters that minimise the \emph{expected} cost, with an
excess of order $\sigma^{2}p/n$ by the variance formula
(\ref{eq:2-olsvariance}); optimising the empirical cost beyond that accuracy
buys nothing, because the next data set would move the minimum by more than
that anyway.  Setting $\epsilon\sim\sigma^{2}p/n$ gives
\begin{equation}
  \text{gradient descent: } \bigO\!\bigl(np\,\kappa\ln n\bigr),
  \qquad
  \text{SGD: } \bigO\!\Bigl(Mp\cdot\frac{n}{\sigma^{2}p}\Bigr)=\bigO\!\bigl(nM/\sigma^{2}\bigr),
  \label{eq:4-flopcompare}
\end{equation}
so that SGD wins by a factor of order $p\kappa\ln n/M$, which for
$\kappa=10^{2}$, $p=10^{3}$ and $M=32$ is several thousand, and grows with
the size of the model and the conditioning of the problem.  This argument,
due to Bottou and Bousquet~\cite{bottou2008}, is the reason the deep
learning frameworks are built around minibatches: for large $n$ the
statistical accuracy that is worth having is reached after a small number of
passes over the data, and in that regime SGD is not merely cheaper per step
but cheaper to the goal.

\paragraph{An example where SGD beats gradient descent.}
Figure~\ref{fig:sgdvsgd} shows the argument at work on a least-squares
problem with $n=10^{5}$ observations and $p=20$ standardised but strongly
correlated features (an autoregressive correlation with coefficient $0.9$
between neighbouring columns, so that $\kappa=213$), unit noise, and
$\bm{\theta}_0=\bm{0}$.  The horizontal axis is the number of floating-point
operations by Eq.~(\ref{eq:4-flops}); the vertical axis is the excess cost
$C(\bm{\theta})-C(\hat{\bm{\theta}})$, and the dotted line is the estimation
error $\sigma^{2}p/n=2\times10^{-4}$ below which further optimisation is
statistically meaningless (the true parameters sit at $2.6\times10^{-4}$).
Gradient descent at $\gamma^{*}$ needs $640$ iterations, $5\times10^{9}$
flops, to reach that line; momentum with the tuned $\gamma$ and $\beta$ of
Eq.~(\ref{eq:4-momentumrate}) needs $81$; SGD with $M=32$ and the schedule
$\gamma_t=20/(t+1000)$ of Eq.~(\ref{eq:4-timedecay}) reaches it after about
five epochs, $4\times10^{7}$ flops -- two orders of magnitude fewer than
gradient descent and one fewer than momentum -- and is within a factor twenty
of it after a single epoch, before gradient descent has completed its
\emph{first} step.  The constant-learning-rate runs make the other point of
this section: they are as fast at the start and then level off, at
$10^{-2}$ for $\gamma=0.02$ and $3\times10^{-3}$ for $\gamma=0.005$, the
signature of the noise that does not decay.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{BookFigures/chapter04_optimization/sgd_vs_gd}
\caption{Excess cost against floating-point operations on a least-squares problem with $n=10^{5}$, $p=20$ and $\kappa=213$, from $\bm{\theta}_0=\bm{0}$.  Full-batch gradient descent at $\gamma^{*}$ and momentum with tuned parameters against minibatch SGD ($M=32$) at two constant learning rates and with the schedule (\ref{eq:4-timedecay}).  The dotted line is the estimation error $\sigma^{2}p/n$ of Eq.~(\ref{eq:4-flopcompare}); SGD with the schedule reaches it in five epochs, momentum in $81$ full gradients and gradient descent in $640$.}
\label{fig:sgdvsgd}
\end{figure}

\begin{Python}{}
import numpy as np

n, p, rho, sigma = 100_000, 20, 0.9, 1.0
rng = np.random.default_rng(2026)
z = rng.normal(size=(n, p))
X = np.empty_like(z)
X[:, 0] = z[:, 0]
for j in range(1, p):                     # AR(1)-correlated columns: kappa = 213 after standardising
    X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * z[:, j]
X = (X - X.mean(axis=0)) / X.std(axis=0)
theta_true = rng.normal(size=p)
y = X @ theta_true + sigma * rng.normal(size=n)

theta_hat = np.linalg.solve(X.T @ X, X.T @ y)          # np^2 + p^3/3 flops
c_hat = np.mean((X @ theta_hat - y)**2)
excess = lambda th: np.mean((X @ th - y)**2) - c_hat
eigs = np.linalg.eigvalsh((2.0 / n) * X.T @ X)
gamma_star = 2.0 / (eigs.max() + eigs.min())
flops_per_point = 4 * p                                # Eq. (4.flops)

theta = np.zeros(p)                                    # gradient descent: 4np flops per step
for k in range(1, 1001):
    theta -= gamma_star * (2.0 / n) * X.T @ (X @ theta - y)
    if excess(theta) < sigma**2 * p / n:
        print(f"gradient descent reaches sigma^2 p/n after {k} steps = {k * n * flops_per_point:.1e} flops")
        break

M, t, theta = 32, 0, np.zeros(p)                       # SGD: 4Mp flops per update, 4np per epoch
for epoch in range(1, 6):
    for b in np.array_split(rng.permutation(n), n // M):
        t += 1
        gamma = 20.0 / (t + 1000.0)                        # Eq. (4.timedecay)
        theta -= gamma * (2.0 / len(b)) * X[b].T @ (X[b] @ theta - y[b])
    print(f"SGD epoch {epoch}: excess cost {excess(theta):.2e} after {t * M * flops_per_point:.1e} flops")
\end{Python}
''')

# 4.7.1: equilibration derivation before the schedule
rep(r'''\index{learning rate!schedule}
Because the gradient noise does not decay, a constant learning rate leaves the
iterate bouncing around the minimum forever.  The standard remedy is to let
$\gamma$ decay with time.''', r'''\index{learning rate!schedule}
Because the gradient noise does not decay, a constant learning rate leaves the
iterate bouncing around the minimum forever.  This can be quantified.  Take
a single eigendirection of a quadratic cost, with curvature $\lambda$, and
write the minibatch gradient as the exact gradient $\lambda e_t$ plus a noise
term $\xi_t$ of mean zero and variance $\sigma_g^{2}/M$ by
Eq.~(\ref{eq:4-minibatchvar}), independent from step to step.  The error
then obeys
\begin{equation}
  e_{t+1}=(1-\gamma\lambda)\,e_t-\gamma\,\xi_t ,
  \label{eq:4-noisyrecursion}
\end{equation}
the recursion~(\ref{eq:4-decoupled}) driven by noise.  Its mean contracts as
before, but its variance settles at the fixed point of
$\mathrm{Var}[e_{t+1}]=(1-\gamma\lambda)^{2}\mathrm{Var}[e_t]+\gamma^{2}\sigma_g^{2}/M$,
\begin{equation}
  \mathrm{Var}[e_\infty]=\frac{\gamma\,\sigma_g^{2}/M}{\lambda\,(2-\gamma\lambda)}
  \approx\frac{\gamma\,\sigma_g^{2}}{2\lambda M}
  \qquad(\gamma\lambda\ll1).
  \label{eq:4-sgdcloud}
\end{equation}
The iterate does not converge; it \emph{equilibrates}, in a cloud around the
minimum whose radius scales as $\sqrt{\gamma/M}$ -- largest along the flat
directions, where $\lambda$ is small -- and which no number of further
iterations will shrink.  This is the plateau seen in
Fig.~\ref{fig:sgdvsgd}, and it says exactly what the remedies are: a smaller
$\gamma$ (at the price of slower progress early on), a larger $M$ (at the
price of $M$ per step), or a $\gamma$ that starts large and ends small.  The
standard remedy is the last: let $\gamma$ decay with time.''')

rep(r'''with $t_0,t_1>0$ fixed.  The learning rate starts at $t_0/t_1$ and decays
towards zero, so that the iterate eventually stops moving.  The classical
conditions for convergence, due to Robbins and Monro, are that
$\sum_t\gamma_t=\infty$ -- so that the iterate can still reach the minimum from
any starting point -- while $\sum_t\gamma_t^{2}<\infty$, so that the accumulated
noise is finite.  Equation~(\ref{eq:4-timedecay}) satisfies both.''',
r'''with $t_0,t_1>0$ fixed.  The learning rate starts at $t_0/t_1$ and decays
towards zero, so that the iterate eventually stops moving.  The classical
conditions for convergence, due to Robbins and Monro~\cite{robbins1951}, are
that $\sum_t\gamma_t=\infty$ -- so that the iterate can still reach the
minimum from any starting point -- while $\sum_t\gamma_t^{2}<\infty$, so that
the accumulated noise, which by Eq.~(\ref{eq:4-noisyrecursion}) enters with
weight $\gamma_t^{2}$, is finite.  Equation~(\ref{eq:4-timedecay}) satisfies
both; a constant satisfies neither.  In a finite run the first condition
bites in practice as well as in principle: with $t_1$ small the learning rate
has collapsed after a few hundred updates and the accumulated step
$\sum_t\gamma_t$ over the run is too short to cross an ill-conditioned
valley, so the iterate freezes far from the minimum.  The cure is to keep
the initial rate $t_0/t_1$ and enlarge both constants, which delays the
decay; the schedule in Fig.~\ref{fig:sgdvsgd} uses $t_0=20$, $t_1=1000$.''')

# ---------------------------------------------------------------- 4.8 why adapt: preconditioning view
rep(r'''without ever forming a second derivative, and dividing by its square root
approximates the $\bm{H}^{-1/2}$ scaling.  This is the entire principle behind
AdaGrad, RMSProp and Adam; the three differ only in how the average is taken.
''', r'''without ever forming a second derivative, and dividing by its square root
approximates the $\bm{H}^{-1/2}$ scaling.  This is the entire principle behind
AdaGrad, RMSProp and Adam; the three differ only in how the average is taken.

\paragraph{The idea made precise: diagonal preconditioning.}
All three methods are instances of the \emph{preconditioned} iteration
\begin{equation}
  \bm{\theta}_{t+1}=\bm{\theta}_t-\gamma\,\bm{D}_t^{-1}\nabla C(\bm{\theta}_t),
  \qquad
  \bm{D}_t=\mathrm{diag}\bigl(\sqrt{v_{t,1}},\dots,\sqrt{v_{t,p}}\bigr),
  \label{eq:4-precond}
\end{equation}
with Newton's method the case $\bm{D}=\bm{H}$ and gradient descent the case
$\bm{D}=\bm{I}$.  For a quadratic cost the error recursion becomes
$\bm{e}_{t+1}=(\bm{I}-\gamma\bm{D}^{-1}\bm{H})\bm{e}_t$, so what governs
stability and speed is no longer the spectrum of $\bm{H}$ but that of
$\bm{D}^{-1}\bm{H}$.  When $\bm{H}$ is diagonal and $\bm{D}$ tracks its
diagonal, $\bm{D}^{-1}\bm{H}=\bm{I}$: every direction contracts at the same
rate, the condition number has been reduced to one, and the iteration count
no longer depends on $\kappa$ at all.  When $\bm{H}$ has off-diagonal
entries, no diagonal $\bm{D}$ can achieve this; the best a diagonal
preconditioner can do is bounded by how far $\bm{H}$ is from diagonal, and a
valley that runs at $45^\circ$ to the axes, with the two coordinates equally
scaled, is not improved at all.  Figure~\ref{fig:adaptivepaths} shows both
cases on the bowl with eigenvalues $1$ and $100$: on the axis-aligned bowl the
adaptive methods set off diagonally, both coordinates moving at about
$\gamma$ per step, and reach $\|\bm{\theta}\|<10^{-4}$ in $297$ (AdaGrad),
$52$ (RMSProp) and $151$ (Adam) iterations against $546$ for gradient descent
at $0.9\cdot2/\lambda_{\max}$ and $162$ for momentum; on the same bowl rotated
by $45^\circ$ gradient descent and momentum are \emph{unchanged}, because
Eq.~(\ref{eq:4-decoupled}) depends only on the eigenvalues, while AdaGrad and
RMSProp no longer reach $10^{-4}$ in $5000$ iterations and Adam needs $635$.
The adaptive methods see the diagonal of the Hessian and nothing else.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.98\textwidth]{BookFigures/chapter04_optimization/adaptive_paths}
\caption{The first hundred steps of five optimisers on the quadratic bowl with Hessian eigenvalues $1$ and $100$, from the same starting point.  Left: the Hessian is diagonal, and the adaptive methods, which rescale each coordinate by its own gradient history, move diagonally towards the minimum while plain gradient descent zigzags down the steep direction and crawls along the flat one.  Right: the same bowl rotated by $45^\circ$, with the same eigenvalues and condition number.  Gradient descent and momentum are unaffected; the adaptive methods, whose diagonal rescaling cannot see the off-diagonal curvature, lose most of their advantage.}
\label{fig:adaptivepaths}
\end{figure}

\paragraph{Scale invariance.}
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
of reach.
''')

# ---------------------------------------------------------------- 4.9 AdaGrad: effective schedule
rep(r'''In convex optimisation AdaGrad achieves a convergence rate comparable to the
best fixed learning rate tuned in hindsight for the problem, which is a strong
guarantee and effectively removes the need to tune $\gamma$ by hand.
''', r'''In convex optimisation AdaGrad achieves a convergence rate comparable to the
best fixed learning rate tuned in hindsight for the problem, which is a strong
guarantee and effectively removes the need to tune $\gamma$ by hand.

\paragraph{The built-in schedule.}
The form of the decay is easy to read off.  Along a direction where the
gradient keeps a roughly constant magnitude $|g|$ -- a flat direction far
from the minimum -- the accumulator is $r_{t}\approx t\,g^{2}$ and the
effective learning rate is
\begin{equation}
  \alpha_{t}=\frac{\gamma}{\sqrt{\epsilon+r_{t}}}\approx\frac{\gamma}{|g|\sqrt{t}},
  \label{eq:4-adagradschedule}
\end{equation}
a $1/\sqrt{t}$ schedule of exactly the kind that Section~\ref{sec:schedules}
introduces by hand, but scaled by the size of the gradient in that
coordinate, so that every direction moves by about $\gamma/\sqrt{t}$
regardless of how steep it is.  At the very first step $r_1=g_1^{2}$ and the
update is $\gamma\,\mathrm{sign}(g_1)$ in every coordinate: the method
starts by moving each parameter by the same distance $\gamma$, whatever the
gradient.  This is the sense in which $\gamma$ is a \emph{length} for the
adaptive methods rather than a multiplier of the gradient, a point that
Section~\ref{sec:adaptivecode} returns to.  The cumulative displacement along
the constant-gradient direction is $\sum_t\gamma/\sqrt{t}\sim2\gamma\sqrt{t}$,
which grows without bound but ever more slowly: AdaGrad can reach any
minimum in principle, and takes increasingly long to do so in practice.
''')

# ---------------------------------------------------------------- 4.10 RMSProp: cannot converge at constant gamma
rep(r'''RMSProp is thus the same idea as AdaGrad with a finite memory, and the
resemblance to the momentum update~(\ref{eq:4-momentum}) is not accidental.
Both are exponential moving averages; momentum averages the gradient, RMSProp
averages its square.  It is natural to ask what happens if one does both.
''', r'''RMSProp is thus the same idea as AdaGrad with a finite memory, and the
resemblance to the momentum update~(\ref{eq:4-momentum}) is not accidental.
Both are exponential moving averages; momentum averages the gradient, RMSProp
averages its square.  It is natural to ask what happens if one does both.

\paragraph{The price of forgetting.}
Finite memory removes AdaGrad's decay, and with it AdaGrad's ability to
settle.  Consider the one-dimensional quadratic $C=\frac12\lambda\theta^{2}$
with a constant learning rate.  Once the iterate is close to the minimum the
gradient $g_t=\lambda\theta_t$ is small and $v_t$, an average of recent
$g^{2}$, is small with it: for a slowly varying iterate $v_t\approx g_t^{2}$,
and the update~(\ref{eq:4-rmsprop}) becomes
\begin{equation}
  \theta_{t+1}\approx\theta_t-\gamma\,\mathrm{sign}(g_t)
  \qquad\text{whenever}\qquad g_t^{2}\gg\epsilon ,
  \label{eq:4-rmspropsign}
\end{equation}
a step of fixed length $\gamma$ that cannot shrink.  Convergence would
require $|\theta_{t+1}|<|\theta_t|$ for all large $t$, that is
$\gamma\lambda/\sqrt{v_t+\epsilon}<2$; but as $\theta_t\to0$ the
denominator tends to $\sqrt{\epsilon}$, and the condition becomes
$\gamma\lambda<2\sqrt{\epsilon}$, which with $\epsilon=10^{-8}$ means
$\gamma\lambda<2\times10^{-4}$ -- a learning rate too small to be useful.
For any practical $\gamma$ the iterate therefore ends in a limit cycle of
amplitude of order $\gamma$ around the minimum, regardless of $\lambda$ and
of $\rho$, and the excess cost floors at about $\lambda\gamma^{2}/2$.  On the
convex polynomial fits of Section~\ref{sec:gdlinreg} this is exactly what
one observes: RMSProp at a constant $\gamma$ reaches an excess cost of
$10^{-4}$ and never $10^{-6}$, at every $\gamma$ between $10^{-3}$ and $1$,
while AdaGrad, whose built-in $1/\sqrt{t}$ decay of
Eq.~(\ref{eq:4-adagradschedule}) is precisely what has been discarded,
converges to rounding.  RMSProp needs a decaying $\gamma$ of the kind of
Section~\ref{sec:schedules} to converge at all; in the non-convex, noisy
setting of deep learning, where nobody expects to sit at a minimum, this is a
price cheerfully paid for not stalling.
''')

# ---------------------------------------------------------------- 4.11 Adam: size of the step, settling, caveat
rep(r'''\paragraph{Adam against its predecessors.}
AdaGrad uses per-coordinate scaling like Adam but has no momentum, and slows
down excessively because its accumulation never forgets.  RMSProp uses a
moving average of squared gradients, so it does not slow down, but includes
neither momentum nor bias correction.  Adam is, in effect, RMSProp plus
momentum plus bias correction: the first moment provides acceleration and
smoother convergence, the second moderates the step size per dimension, and
the correction ensures the estimates are sound from the first iteration.
''', r'''\paragraph{The size of an Adam step.}
Two properties of the update~(\ref{eq:4-adam}) follow from the algebra and
explain much of the method's behaviour in practice.  First, at $t=1$ the
bias-corrected moments are $\hat{\bm{m}}_1=\bm{g}_1$ and
$\hat{\bm{v}}_1=\bm{g}_1^{2}$, so the first step is
$\Delta\bm{\theta}_1=-\alpha\,\mathrm{sign}(\bm{g}_1)$ (up to $\epsilon$): a
move of length $\alpha$ in every coordinate, whatever the gradient, as for
AdaGrad and RMSProp.  Without the correction the first step would be
$\alpha(1-\beta_1)/\sqrt{1-\beta_2}\approx3.2\alpha$ with the default
constants -- the raw first moment is too small by a factor $10$, the raw
second moment by $10^{3}$, and the ratio inherits the mismatch.  Second, at
every step the magnitude of the update is bounded: since $\hat{\bm{m}}_t$ is
a weighted average of the gradients $\bm{g}_1,\dots,\bm{g}_t$ and
$\hat{\bm{v}}_t$ a weighted average of their squares, the ratio
$|\hat m_{t,j}|/\sqrt{\hat v_{t,j}}$ is at most $(1-\beta_1)/\sqrt{1-\beta_2}$
in the worst case and at most one in the common case where the two sets of
weights are comparable, so
\begin{equation}
  |\Delta\theta_{t,j}|\;\lesssim\;\alpha .
  \label{eq:4-adambound}
\end{equation}
The learning rate of Adam is a \emph{trust region}: the largest distance any
parameter moves in one iteration, in the units of that parameter.  This is
why the defaults transfer between problems as gradient descent's $\gamma$
never does, why Adam cannot diverge in the sense of Eq.~(\ref{eq:4-stability})
however $\alpha$ is chosen (on the polynomial fit of
Section~\ref{sec:gdlinreg} it converges for every $\alpha$ between $10^{-3}$
and $1$, with the best value in the middle), and why its default
$\alpha=10^{-3}$ is a hundred times smaller than a typical gradient descent
learning rate.

The first moment is also what lets Adam settle where RMSProp cannot.  Near a
minimum the gradient alternates in sign from step to step, so its average
$\hat{\bm{m}}_t$ is much smaller than its typical magnitude
$\sqrt{\hat{\bm{v}}_t}$, and the ratio in Eq.~(\ref{eq:4-adam}) goes to zero
even at a constant $\alpha$; the fixed-length step of
Eq.~(\ref{eq:4-rmspropsign}) has been averaged away.  On the convex problems
of this chapter Adam therefore converges to rounding while RMSProp rattles.
It should be added that the convergence proof in the original paper contains
an error, and that Reddi, Kale and Kumar~\cite{reddi2018} constructed simple
convex problems on which Adam does not converge, because the effective
learning rate $\alpha/\sqrt{\hat{\bm{v}}_t}$ can \emph{increase} when a
large gradient is forgotten; their repair, AMSGrad, keeps the running
maximum of $\hat{\bm{v}}_t$ instead.  In practice the failure is rare and
Adam as stated remains the default.

\paragraph{Adam against its predecessors.}
AdaGrad uses per-coordinate scaling like Adam but has no momentum, and slows
down excessively because its accumulation never forgets.  RMSProp uses a
moving average of squared gradients, so it does not slow down, but includes
neither momentum nor bias correction, and by Eq.~(\ref{eq:4-rmspropsign})
cannot settle at a constant learning rate.  Adam is, in effect, RMSProp plus
momentum plus bias correction: the first moment provides acceleration,
smoother convergence and the ability to settle, the second moderates the step
size per dimension, and the correction ensures the estimates are sound from
the first iteration.
''')

# ---------------------------------------------------------------- 4.12: the step-function form
rep(r'''Running all five on the data of Section~\ref{sec:gdlinreg} for a hundred
epochs with batches of ten, for three values of the learning rate, gives the
final cost values of Table~\ref{tab:optimcompare}.''', r'''The same five updates can be written as a single \emph{step function} that
knows nothing about data, epochs or minibatches: it receives a gradient and
returns the new parameters, keeping its running quantities in a dictionary.
The data loop of the previous listing, or the full-gradient loop of
Section~\ref{sec:gdlinreg}, then calls it once per gradient.  This is the
structure we recommend for the projects, because it separates the three
things that can go wrong -- the cost, its gradient, and the optimiser -- and
lets an analytical gradient and \texttt{jax.grad} of Section~\ref{sec:autodiff}
be exchanged without touching the optimiser.

\begin{Python}{}
import numpy as np

def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t = 1, 2, ...; state carries the running quantities."""
    if method == "plain":                                                       # Eq. (4.gd)
        return theta - gamma * g, state
    if method == "momentum":                                                    # Eq. (4.momentum)
        state["v"] = v = beta * state.get("v", 0.0) + gamma * g
        return theta - v, state
    if method == "adagrad":                                                     # Eqs. (4.adagradaccum), (4.adagrad)
        state["r"] = r = state.get("r", 0.0) + g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "rmsprop":                                                     # Eqs. (4.rmspropaccum), (4.rmsprop)
        state["r"] = r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "adam":                                                        # Eqs. (4.adamfirst)-(4.adam)
        state["m"] = m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g
        state["r"] = r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g
        m_hat, r_hat = m / (1.0 - beta1**t), r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters=1000, tol=1e-8, **kw):
    """Full-gradient loop around optimiser_step; returns every iterate."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        if np.linalg.norm(g) < tol:
            break
    return np.array(history)
\end{Python}

Running all five on the data of Section~\ref{sec:gdlinreg} for a hundred
epochs with batches of ten, for three values of the learning rate, gives the
final cost values of Table~\ref{tab:optimcompare}.''')

# ---------------------------------------------------------------- summary paragraph: SGD and adaptive sentences
rep(r'''$1/(1-\beta)$ and cancelling oscillatory ones, which improves the rate from
$\kappa$ to $\sqrt{\kappa}$.  Stochastic gradients replace the full sum by a
minibatch, trading a worse rate per iteration for a far better rate per unit
of computation, and adding noise that helps escape poor local minima.  The
adaptive methods build a diagonal estimate of curvature from the second moment
of the gradient: AdaGrad by an unbounded sum, which eventually stalls; RMSProp
by an exponential moving average, which does not; and Adam by combining
RMSProp with momentum and correcting the initialisation bias with the factors
$1-\beta_i^{t}$ derived in Eq.~(\ref{eq:4-adambiasderiv}).''',
r'''$1/(1-\beta)$ and cancelling oscillatory ones, which improves the rate from
$\kappa$ to $\sqrt{\kappa}$.  Stochastic gradients replace the full sum by a
minibatch -- an unbiased estimate with noise $\propto1/\sqrt{M}$,
Eq.~(\ref{eq:4-minibatchvar}) -- trading a worse rate per iteration for a
far better rate per floating-point operation: by the count of
Eq.~(\ref{eq:4-flopcompare}) the statistical accuracy worth having is
reached in a few passes over the data, where gradient descent needs
$\kappa\ln n$ of them, and Fig.~\ref{fig:sgdvsgd} showed the factor to be
two orders of magnitude on an ordinary problem.  The price is that at a
constant learning rate the iterate equilibrates in a cloud of radius
$\propto\sqrt{\gamma/M}$, Eq.~(\ref{eq:4-sgdcloud}), instead of converging,
which the Robbins--Monro schedules repair.  The adaptive methods are
diagonal preconditioners, Eq.~(\ref{eq:4-precond}), that estimate the
curvature from the second moment of the gradient: AdaGrad by an unbounded
sum, which gives a built-in $1/\sqrt{t}$ schedule and eventually stalls;
RMSProp by an exponential moving average, which does not stall but cannot
settle at a constant learning rate, Eq.~(\ref{eq:4-rmspropsign}); and Adam
by combining RMSProp with momentum, whose averaging restores the ability to
settle, and correcting the initialisation bias with the factors
$1-\beta_i^{t}$ derived in Eq.~(\ref{eq:4-adambiasderiv}).  Their learning
rate is a trust region, Eq.~(\ref{eq:4-adambound}), not a multiplier of the
gradient; they are invariant to the scaling of individual features and blind
to their correlation, Fig.~\ref{fig:adaptivepaths}.''')

# programs list
rep(r'''  \item \texttt{stochastic\_gradient.py} -- minibatch SGD with a varying number
        of batches, the time-decay schedule~(\ref{eq:4-timedecay}), and a
        comparison of convergence per iteration against convergence per unit
        of computation.''', r'''  \item \texttt{stochastic\_gradient.py} -- minibatch SGD with a varying number
        of batches, the time-decay schedule~(\ref{eq:4-timedecay}), the
        large-$n$ comparison of Fig.~\ref{fig:sgdvsgd} counted in
        floating-point operations, and the empirical check of the minibatch
        variance~(\ref{eq:4-minibatchvar}) and of the equilibration
        radius~(\ref{eq:4-sgdcloud}).''')
rep(r'''  \item \texttt{adaptive\_optimizers.py} -- AdaGrad, RMSProp and Adam as in
        Section~\ref{sec:adaptivecode}, the learning-rate sweep of
        Table~\ref{tab:optimcompare}, and the same comparison repeated on a
        deliberately ill-conditioned design matrix where the adaptive methods
        win decisively.''', r'''  \item \texttt{adaptive\_optimizers.py} -- AdaGrad, RMSProp and Adam as in
        Section~\ref{sec:adaptivecode}, both as the epoch loop and as the
        step function \texttt{optimiser\_step}, the learning-rate sweep of
        Table~\ref{tab:optimcompare}, the axis-aligned and rotated bowls of
        Fig.~\ref{fig:adaptivepaths}, and the same comparison repeated on a
        deliberately ill-conditioned design matrix.''')

# ---------------------------------------------------------------- exercises: three new warm-ups before the FD one
rep(r'''\item \textbf{The finite-difference error curve (numerical).}
  Reproduce Fig.~\ref{fig:4-fdvsad} for $f(x)=e^{x}$ at $x=1$ and for''', r'''\item \textbf{The minibatch gradient as an estimator.}
  (a) Derive Eq.~(\ref{eq:4-minibatchvar}) for sampling without replacement,
      starting from the fact that each example enters the batch with
      probability $M/n$ and each pair with probability $M(M-1)/(n(n-1))$.
  (b) On the least-squares problem of Section~\ref{sec:gdlinreg}, fix a point
      $\bm{\theta}$, draw a few hundred minibatches for $M=1,5,25$ and compare
      the mean and the standard deviation of the minibatch gradients with the
      full gradient and with Eq.~(\ref{eq:4-minibatchvar}).
  (c) Run SGD at a constant $\gamma$ for $M=5$ and $M=20$, measure the size of
      the cloud in which the iterate settles along the flattest
      eigendirection, and compare with Eq.~(\ref{eq:4-sgdcloud}).

\item \textbf{Counting flops.}
  (a) Verify the count~(\ref{eq:4-flops}) and extend it to the Ridge
      gradient~(\ref{eq:4-ridgegradhess}) and to one Newton step with the
      Hessian~(\ref{eq:4-gdhessian}) formed and factorised.
  (b) For $n=10^{6}$, $p=10^{3}$, $\kappa=10^{3}$ and $M=64$, estimate from
      Eq.~(\ref{eq:4-flopcompare}) the number of floating-point operations
      gradient descent, momentum and SGD need to reach the estimation error
      $\sigma^{2}p/n$, and the number of epochs each corresponds to.
  (c) Repeat the experiment of Fig.~\ref{fig:sgdvsgd} with $n=10^{3}$ and
      with $n=10^{6}$.  Where does the crossover at which SGD overtakes
      gradient descent move, and why?

\item \textbf{RMSProp cannot settle; Adam can.}
  (a) For $C=\frac12\lambda\theta^{2}$ with $\lambda=1$, run RMSProp at
      $\gamma=10^{-2}$ from $\theta_0=1$ for $10^{4}$ steps and plot
      $|\theta_t|$.  Confirm the limit cycle predicted by
      Eq.~(\ref{eq:4-rmspropsign}) and measure its amplitude for
      $\gamma=10^{-1},10^{-2},10^{-3}$.
  (b) Repeat with Adam at the same $\gamma$ and explain, using the first
      moment, why it converges.
  (c) Show from Eqs.~(\ref{eq:4-adamfirst})--(\ref{eq:4-adam}) that the
      first Adam step is $-\alpha\,\mathrm{sign}(\bm{g}_1)$, and that without
      bias correction it would be longer by $(1-\beta_1)/\sqrt{1-\beta_2}$.
  (d) Multiply one column of the design matrix of
      Section~\ref{sec:gdlinreg} by $c=100$ and rerun gradient descent, Adam
      and AdaGrad from the same starting point.  Which iteration counts
      change, and why?

\item \textbf{The finite-difference error curve (numerical).}
  Reproduce Fig.~\ref{fig:4-fdvsad} for $f(x)=e^{x}$ at $x=1$ and for''')

assert s != orig
open("chapter4.tex", "w").write(s)
print("patched; figures:", s.count(r"\begin{figure}"), "python blocks:", s.count(r"\begin{Python}"))
