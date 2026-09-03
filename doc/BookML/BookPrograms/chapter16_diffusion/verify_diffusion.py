"""Chapter 16: every verification quoted in Sections 16.2 and 16.5.1."""
import autograd.numpy as np
from autograd import grad as ag, jacobian
import diffusion as D
import flows

T = 200
beta, alpha, abar = D.linear_schedule(T)
rng = np.random.default_rng(0)

print("=== 1. the closed-form marginal vs simulating the chain ===")
print("   t    simulated mean/sd of x_t      closed form sqrt(abar), sqrt(1-abar)")
x = np.full((20000, 1), 2.0)
for i in range(T):
    x = np.sqrt(alpha[i]) * x + np.sqrt(beta[i]) * rng.normal(size=x.shape)
    if i in (9, 49, 99, 199):
        print(f"  {i+1:3d}    {x.mean():8.4f} / {x.std():7.4f}          "
              f"{2*np.sqrt(abar[i]):8.4f} / {np.sqrt(1-abar[i]):7.4f}")

print("\n=== 2. the forward posterior vs Bayes by quadrature (1-D) ===")
print("    t     quadrature mean / var        closed form mean / var")
x0v, xtv = 1.7, 0.9
g = np.linspace(-8, 8, 400001)
for i in [5, 50, 150, 199]:
    lp1 = -0.5 * (xtv - np.sqrt(alpha[i]) * g) ** 2 / beta[i]
    ab_prev = abar[i-1] if i > 0 else 1.0
    lp2 = -0.5 * (g - np.sqrt(ab_prev) * x0v) ** 2 / (1 - ab_prev)
    w = np.exp(lp1 + lp2 - np.max(lp1 + lp2)); w /= w.sum()
    m = float(np.sum(w * g)); v = float(np.sum(w * (g - m) ** 2))
    mu, var = D.posterior(np.array([[xtv]]), np.array([[x0v]]), np.array([i]),
                          beta, alpha, abar)
    print(f"  {i:4d}     {m:9.6f} / {v:.3e}       "
          f"{float(mu[0,0]):9.6f} / {float(var[0,0]):.3e}")

print("\n=== 3. the score of the forward marginal is the negative noise ===")
print("     t     -eps/sqrt(1-abar_t)      d/dx_t log q(x_t|x_0)   |diff|")
for i in [10, 80, 199]:
    eps = 0.63
    xt = np.sqrt(abar[i]) * x0v + np.sqrt(1 - abar[i]) * eps
    f = lambda z: -0.5 * (z - np.sqrt(abar[i]) * x0v) ** 2 / (1 - abar[i])
    a = -eps / np.sqrt(1 - abar[i]); b = float(ag(f)(xt))
    print(f"  {i:4d}     {a:14.8f}     {b:19.8f}   {abs(a-b):.2e}")

# --- the flow, Section 16.5.1 ----------------------------------------------
print("\n=== 1. the analytic log-det vs the numerical Jacobian determinant ===")
r = np.random.default_rng(0)
d = 4
L = flows.init_coupling(d, hidden=16, n_layers=4, rng=r)
z = r.normal(size=(3, d))
x, ld = flows.forward(L, z)
for i in range(3):
    J = jacobian(lambda zz: flows.forward(L, zz[None, :])[0][0])(z[i])
    num = np.log(abs(np.linalg.det(J)))
    print(f"  sample {i}: analytic {ld[i]:.10f}   numerical {num:.10f}   "
          f"diff {abs(ld[i]-num):.2e}")

print("\n=== 2. the flow is exactly invertible ===")
zz, _ = flows.inverse(L, x)
print(f"  max|f^-1(f(z)) - z| = {np.abs(zz-z).max():.2e}")

print("\n=== 3. the density integrates to one (2-D, quadrature) ===")
L2 = flows.init_coupling(2, hidden=16, n_layers=4, rng=np.random.default_rng(1))
g = np.linspace(-8, 8, 701); dg = g[1] - g[0]
G1, G2 = np.meshgrid(g, g, indexing="ij")
P = np.exp(flows.log_prob(L2, np.column_stack([G1.ravel(), G2.ravel()])))
print(f"  integral of p(x) dx = {P.sum()*dg*dg:.8f}")
