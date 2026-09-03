"""The eight checks of Chapter 17.

Nothing here is quoted from a paper: every identity used in the chapter is
recomputed, either in closed form against quadrature or by training the network
the theorem is about.
"""
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize_scalar
import gan

out = open("verify.txt", "w", buffering=1)
rng = np.random.default_rng(0)

gauss = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
x = np.linspace(-8.0, 10.0, 200001)
dx = x[1] - x[0]

# ---------------------------------------------------------------------------
# 1.  the optimal discriminator, pointwise
# ---------------------------------------------------------------------------
out.write("=== 1. the pointwise maximiser of A log t + B log(1-t) ===\n")
out.write("      A          B        numerical t*       A/(A+B)      |diff|\n")
worst = 0.0
for A, B in [(0.9, 0.1), (0.5, 0.5), (0.2, 0.8), (1e-3, 0.4), (0.37, 0.11)]:
    f = lambda t: -(A * np.log(t) + B * np.log1p(-t))
    t_num = minimize_scalar(f, bounds=(1e-12, 1 - 1e-12), method="bounded",
                            options={"xatol": 1e-14}).x
    t_closed = A / (A + B)
    worst = max(worst, abs(t_num - t_closed))
    out.write(f"  {A:8.4f}   {B:8.4f}   {t_num:14.10f}  {t_closed:12.10f}  "
              f"{abs(t_num-t_closed):8.1e}\n")
out.write(f"  worst deviation from Eq. (17.dstar): {worst:.2e}\n\n")

# second derivative test
A, B, t = 0.37, 0.11, 0.37 / 0.48
out.write(f"  f''(t*) = {-A/t**2 - B/(1-t)**2:.4f} < 0, so t* is a maximum\n\n")

# ---------------------------------------------------------------------------
# 2.  a trained discriminator reproduces D*
# ---------------------------------------------------------------------------
out.write("=== 2. a trained discriminator against a known pair of densities ===\n")
mr, sr, mg, sg = 0.0, 1.0, 1.5, 0.6
Xr = rng.normal(mr, sr, (20000, 1))
Xg = rng.normal(mg, sg, (20000, 1))
Q = gan.init_mlp((1, 64, 64, 1), rng)
M, V, t_ = gan.adam_state(Q)


def bce(Q, xr, xg):
    return np.mean(gan.softplus(-gan.discriminator(Q, xr))) \
         + np.mean(gan.softplus(gan.discriminator(Q, xg)))


g_bce = grad(bce, 0)
for it in range(4000):
    i = rng.integers(0, len(Xr), 256)
    j = rng.integers(0, len(Xg), 256)
    Q = gan.adam_step(Q, g_bce(Q, Xr[i], Xg[j]), M, V, t_, 2e-3)

grid = np.linspace(-4.0, 5.0, 361)[:, None]
D_net = gan.sigmoid(gan.discriminator(Q, grid))
pr = gauss(grid[:, 0], mr, sr)
pg = gauss(grid[:, 0], mg, sg)
D_star = pr / (pr + pg)
mix = 0.5 * (pr + pg)
where = mix > 0.01 * mix.max()          # where the samples actually are
out.write("     x        mixture     D_net(x)   p_r/(p_r+p_g)   |diff|\n")
for k in range(0, 361, 40):
    out.write(f"  {grid[k,0]:6.2f}  {mix[k]:11.2e}  {D_net[k]:9.5f}  "
              f"{D_star[k]:13.5f}  {abs(D_net[k]-D_star[k]):8.4f}\n")
out.write(f"  where the mixture density exceeds 1% of its peak "
          f"(x in [{grid[where,0].min():.2f},{grid[where,0].max():.2f}]):\n")
out.write(f"      max |D_net - D*| = {np.max(np.abs(D_net-D_star)[where]):.4f}, "
          f"rms = {np.sqrt(np.mean(((D_net-D_star)[where])**2)):.4f}\n")
out.write(f"  over the whole grid, including the empty tails:\n")
out.write(f"      max |D_net - D*| = {np.max(np.abs(D_net-D_star)):.4f}, "
          f"rms = {np.sqrt(np.mean((D_net-D_star)**2)):.4f}\n")

# the same network, read as a value function
xs = rng.normal(mr, sr, (400000, 1))
gs = rng.normal(mg, sg, (400000, 1))
V_net = -np.mean(gan.softplus(-gan.discriminator(Q, xs))) \
        - np.mean(gan.softplus(gan.discriminator(Q, gs)))
prq = gauss(x, mr, sr); pgq = gauss(x, mg, sg); mq = 0.5 * (prq + pgq)
ok = mq > 1e-300
jsq = 0.5 * np.sum(prq[ok] * np.log(prq[ok] / mq[ok]) * dx) \
    + 0.5 * np.sum(pgq[ok] * np.log(pgq[ok] / mq[ok]) * dx)
out.write(f"  V from the trained network : {V_net:.6f}\n")
out.write(f"  2 JS - 2 log 2 by quadrature: {2*jsq-2*np.log(2):.6f}\n\n")


np.save("dstar_grid.npy", grid[:, 0])
np.save("dstar_pr.npy", pr)
np.save("dstar_pg.npy", pg)
np.save("dstar_net.npy", D_net)
np.save("dstar_closed.npy", D_star)

# ---------------------------------------------------------------------------
# 3.  V(G,D*) = 2 JS(p_r||p_g) - 2 log 2
# ---------------------------------------------------------------------------
out.write("=== 3. the value at the optimal discriminator is the JS divergence ===\n")
out.write("   m_g   s_g      V(G,D*)       2 JS - 2log2        |diff|      JS\n")
for mg_, sg_ in [(0.0, 1.0), (0.5, 1.0), (1.5, 0.6), (3.0, 1.0), (6.0, 0.8)]:
    pr = gauss(x, 0.0, 1.0)
    pg = gauss(x, mg_, sg_)
    m = 0.5 * (pr + pg)
    ok = m > 1e-300
    V = np.sum(pr[ok] * np.log(pr[ok] / (pr[ok] + pg[ok])) * dx) \
      + np.sum(pg[ok] * np.log(pg[ok] / (pr[ok] + pg[ok])) * dx)
    js = 0.5 * np.sum(pr[ok] * np.log(pr[ok] / m[ok]) * dx) \
       + 0.5 * np.sum(pg[ok] * np.log(pg[ok] / m[ok]) * dx)
    out.write(f"  {mg_:4.1f}  {sg_:4.1f}  {V:12.8f}  {2*js-2*np.log(2):14.8f}  "
              f"{abs(V-(2*js-2*np.log(2))):10.2e}  {js:8.5f}\n")
out.write(f"  the equal case p_g = p_r gives V = -2log2 = {-2*np.log(2):.8f} "
          f"and D* = 1/2\n")
out.write(f"  the JS divergence is bounded by log 2 = {np.log(2):.8f}\n\n")

# ---------------------------------------------------------------------------
# 4.  one-sided label smoothing
# ---------------------------------------------------------------------------
out.write("=== 4. one-sided label smoothing, D*_s = s p_r/(p_r+p_g) ===\n")
out.write("     s        A        B     numerical t*    s A/(A+B)      |diff|\n")
for s in [1.0, 0.9, 0.7]:
    for A, B in [(0.6, 0.4), (0.25, 0.75)]:
        f = lambda t: -(A * (s * np.log(t) + (1 - s) * np.log1p(-t))
                        + B * np.log1p(-t))
        t_num = minimize_scalar(f, bounds=(1e-12, 1 - 1e-12), method="bounded",
                                options={"xatol": 1e-14}).x
        out.write(f"  {s:5.2f}  {A:7.3f}  {B:7.3f}  {t_num:13.9f}  "
                  f"{s*A/(A+B):11.9f}  {abs(t_num-s*A/(A+B)):9.1e}\n")
out.write("  the smoothed optimum is the unsmoothed one scaled by s: it can no\n"
          "  longer reach 1, so the generator keeps receiving a gradient\n\n")

# ---------------------------------------------------------------------------
# 5.  saturating against non-saturating gradients
# ---------------------------------------------------------------------------
out.write("=== 5. the two generator losses at D(G(z)) = eps ===\n")
out.write("      eps     |dL_sat/du|   |dL_nonsat/du|      ratio    (1-eps)/eps\n")
for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
    u = np.log(eps / (1 - eps))                       # logit with D = eps
    d_sat = grad(lambda u: -gan.softplus(u))(u)       # d/du log(1-D)
    d_non = grad(lambda u: gan.softplus(-u))(u)       # d/du (-log D)
    out.write(f"  {eps:8.1e}  {abs(d_sat):12.3e}  {abs(d_non):14.6f}  "
              f"{abs(d_non/d_sat):10.1f}  {(1-eps)/eps:12.1f}\n")
out.write("  the saturating loss loses its gradient exactly when the generator\n"
          "  is worst; the non-saturating loss gains one\n\n")

# ---------------------------------------------------------------------------
# 6.  where the Jensen-Shannon divergence fails and Wasserstein does not
# ---------------------------------------------------------------------------
out.write("=== 6. JS saturates, W does not: p_r = U[0,1], p_g = U[th,th+1] ===\n")
out.write("     theta        JS(p_r||p_g)        log 2        W(p_r,p_g)\n")
for th in [0.0, 0.25, 0.5, 0.9, 1.0, 2.0, 5.0]:
    ov = max(0.0, 1.0 - th)                # length of the overlap
    # on the overlap both densities are 1, m = 1: contributes 0 to the JS.
    # elsewhere one density is 1 and m = 1/2: contributes log 2 per unit length.
    js = (1.0 - ov) * np.log(2.0)
    out.write(f"  {th:8.2f}  {js:16.8f}  {np.log(2):14.8f}  {th:14.4f}\n")
out.write("  for theta >= 1 the supports are disjoint, JS is pinned at log 2 and\n"
          "  its gradient in theta is zero; W = theta keeps a gradient of one\n\n")

out.write("=== 6b. JS and W for two unit Gaussians, by quadrature ===\n")
out.write("       mu       JS(N(0,1)||N(mu,1))      W = |mu|\n")
for mu in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    pr = gauss(x, 0.0, 1.0)
    pg = gauss(x, mu, 1.0)
    m = 0.5 * (pr + pg)
    ok = m > 1e-300
    js = 0.5 * np.sum(pr[ok] * np.log(pr[ok] / m[ok]) * dx) \
       + 0.5 * np.sum(pg[ok] * np.log(pg[ok] / m[ok]) * dx)
    out.write(f"  {mu:8.2f}  {js:20.8f}  {abs(mu):14.4f}\n")
out.write(f"  JS -> log 2 = {np.log(2):.6f} while W grows without bound\n")
# ---------------------------------------------------------------------------
# 7.  the bilinear game: simultaneous against alternating updates
# ---------------------------------------------------------------------------
out.write("=== 7. the bilinear game V(x,y) = xy, unique Nash at the origin ===\n")
out.write("    eta   steps    ||(x,y)|| simultaneous   (1+eta^2)^(T/2)"
          "    ||(x,y)|| alternating\n")
for eta in [0.1, 0.05, 0.01]:
    T = 1000
    xs, ys = 1.0, 1.0                      # simultaneous
    xa, ya = 1.0, 1.0                      # alternating
    for _ in range(T):
        xs, ys = xs - eta * ys, ys + eta * xs
        xa = xa - eta * ya
        ya = ya + eta * xa                 # uses the *updated* x
    out.write(f"  {eta:5.2f}  {T:6d}   {np.hypot(xs,ys):20.4e}   "
              f"{np.sqrt(2)*(1+eta**2)**(T/2):16.4e}   {np.hypot(xa,ya):18.6f}\n")
out.write("  the simultaneous iteration multiplies the distance to the Nash\n"
          "  equilibrium by sqrt(1+eta^2) at every step, exactly; the alternating\n"
          "  iteration has both eigenvalues on the unit circle and stays bounded\n\n")

# ---------------------------------------------------------------------------
# 8.  why linear interpolation leaves the typical set of the prior
# ---------------------------------------------------------------------------
out.write("=== 8. the norm of a Gaussian latent vector ===\n")
out.write("      k     E||z||    sd(||z||)   E||(z1+z2)/2||   sqrt(k/2)"
          "   deviation in sd\n")
for k in [2, 16, 100]:
    z1 = rng.normal(size=(200000, k))
    z2 = rng.normal(size=(200000, k))
    n1 = np.sqrt(np.sum(z1 ** 2, axis=1))
    nm = np.sqrt(np.sum(((z1 + z2) / 2) ** 2, axis=1))
    out.write(f"  {k:5d}  {n1.mean():9.4f}  {n1.std():10.4f}  {nm.mean():14.4f}"
              f"  {np.sqrt(k/2):10.4f}   {(n1.mean()-nm.mean())/n1.std():13.1f}\n")
out.write("  the midpoint of two prior samples is shorter than a prior sample by\n"
          "  a factor sqrt(2), which at k = 100 is four standard deviations: the\n"
          "  generator has never been trained there.  Hence spherical\n"
          "  interpolation, Eq. (17.slerp)\n\n")


out.close()
print(open("verify.txt").read())
