"""Chapter 15: every verification quoted in Section 15.5."""
import autograd.numpy as np
from scipy.special import logsumexp
import vae

# --- 1. the closed-form KL, Proposition 15.2 --------------------------------
print("=== closed-form Gaussian KL vs Monte Carlo ===")
print("  d_h    closed form    Monte Carlo (10^6)   |diff|")
rng = np.random.default_rng(0)
for dh in [1, 3, 8]:
    mu = rng.normal(size=(1, dh)) * 0.7
    logvar = rng.normal(size=(1, dh)) * 0.5
    kl = float(vae.kl_gaussian(mu, logvar)[0])
    s = np.exp(0.5 * logvar); n = 1000000
    h = mu + s * rng.normal(size=(n, dh))
    lq = np.sum(-0.5*np.log(2*np.pi) - 0.5*logvar - 0.5*((h-mu)/s)**2, axis=1)
    lp = np.sum(-0.5*np.log(2*np.pi) - 0.5*h**2, axis=1)
    mc = float(np.mean(lq - lp))
    print(f"  {dh:3d}    {kl:11.6f}    {mc:14.6f}   {abs(kl-mc):.2e}")

# --- 2. the ELBO bounds the evidence, Theorem 15.1 --------------------------
rng = np.random.default_rng(1)
d, dh = 6, 2
P = vae.init_vae(d, dh, hidden=16, rng=np.random.default_rng(2))
X = (rng.random((5, d)) < 0.5).astype(float)


def log_evidence_grid(P, X, n=401, lim=6.0):
    """log p(x) = log int p(x|h)p(h) dh, by a 2-D quadrature grid."""
    g = np.linspace(-lim, lim, n); dg = g[1] - g[0]
    H1, H2 = np.meshgrid(g, g, indexing="ij")
    H = np.column_stack([H1.ravel(), H2.ravel()])
    lp = np.sum(-0.5*np.log(2*np.pi) - 0.5*H**2, axis=1)
    logits = vae.decode(P, H)
    return np.array([logsumexp(lp + vae.bernoulli_logpdf(
        logits, np.tile(x, (len(H), 1)))) + 2*np.log(dg) for x in X])


le = log_evidence_grid(P, X)
print("\n=== the ELBO is a lower bound on log p(x) ===")
print("  x    log p(x) (quadrature)   ELBO (10^5 samples)      gap")
for i, x in enumerate(X):
    e = rng.normal(size=(100000, dh))
    xb = np.tile(x, (100000, 1))
    mu, lv = vae.encode(P, xb)
    H = mu + np.exp(0.5*lv) * e
    rec = vae.bernoulli_logpdf(vae.decode(P, H), xb)
    el = float(np.mean(rec)) - float(vae.kl_gaussian(mu[:1], lv[:1])[0])
    print(f"  {i}    {le[i]:19.6f}   {el:16.6f}   {le[i]-el:8.6f}")

# --- 3. reparameterised vs score-function gradient, Section 15.4 ------------
c, sigma = 1.3, 1.0
f = lambda h: -(h - c) ** 2
mu0 = 0.4
print("\n=== gradient estimators for E_q[f(h)],  d/dmu ===")
print(f"  exact gradient: {-2*(mu0-c):.6f}\n")
print("   n      reparam mean (sd)          score-function mean (sd)     var ratio")
rng = np.random.default_rng(0)
for n in [10, 100, 1000, 10000]:
    r, s = [], []
    for _ in range(400):
        eps = rng.normal(size=n)
        r.append(np.mean(-2 * (mu0 + sigma*eps - c)))
        h = mu0 + sigma * rng.normal(size=n)
        s.append(np.mean(f(h) * (h - mu0) / sigma**2))
    r, s = np.array(r), np.array(s)
    print(f"  {n:5d}   {r.mean():8.5f} ({r.std():.5f})      "
          f"{s.mean():8.5f} ({s.std():.5f})      {s.var()/r.var():8.1f}")
