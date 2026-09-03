import autograd.numpy as np
from autograd import grad
from nn_de import ACT, init_parameters, network, d_dxk

def pinn_solve(terms, layer_sizes, activation="tanh", n_iter=2000, eta=1e-2,
               rng=None, every=200, verbose=False):
    """terms: list of (name, weight, residual_fn, X). Cost = sum_k w_k * MSE_k."""
    P = init_parameters(layer_sizes, activation, rng)
    def cost(P):
        tot = 0.0
        for _, w, res, Xk in terms:
            tot = tot + w * np.mean(res(P, Xk) ** 2)
        return tot
    gfun = grad(cost)
    b1,b2,eps = 0.9,0.999,1e-8
    m=[[np.zeros_like(W),np.zeros_like(b)] for W,b in P]
    v=[[np.zeros_like(W),np.zeros_like(b)] for W,b in P]
    hist=[]
    for it in range(1, n_iter+1):
        G=gfun(P)
        for l in range(len(P)):
            for j in range(2):
                m[l][j]=b1*m[l][j]+(1-b1)*G[l][j]
                v[l][j]=b2*v[l][j]+(1-b2)*G[l][j]**2
                P[l][j]=P[l][j]-eta*(m[l][j]/(1-b1**it))/(np.sqrt(v[l][j]/(1-b2**it))+eps)
        if it % every == 0 or it == 1:
            parts = {n: float(np.mean(r(P,Xk)**2)) for n,w,r,Xk in terms}
            hist.append((it, float(cost(P)), parts))
            if verbose:
                print(f"  it {it:5d} total {cost(P):.3e}  " +
                      "  ".join(f"{n} {vv:.2e}" for n,vv in parts.items()))
    return P, hist


# ---------------------------------------------------------------------------
# The diffusion and wave equations in the soft (physics-informed) formulation.
# Reproduces Table 9.softvshard of the chapter.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nx = nt = 20
    xs = np.linspace(0, 1, nx); ts = np.linspace(0, 1, nt)
    Xi, Ti = np.meshgrid(xs[1:-1], ts[1:-1], indexing="ij")
    X_col = np.column_stack([Xi.ravel(), Ti.ravel()])
    X_ic = np.column_stack([xs, np.zeros(nx)])
    X_l = np.column_stack([np.zeros(nt), ts])
    X_r = np.column_stack([np.ones(nt), ts])

    def u_net(P, X): return network(P, X, "tanh")
    u_t, u_x = d_dxk(u_net, 1), d_dxk(u_net, 0)
    u_xx, u_tt = d_dxk(u_x, 0), d_dxk(u_t, 1)

    Xf, Tf = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), indexing="ij")
    X_eval = np.column_stack([Xf.ravel(), Tf.ravel()])

    def report(pred, exact, tag):
        e = np.abs(pred - exact)
        m0 = X_eval[:, 1] == 0.0
        print(f"{tag:32s} max {e.max():.3e}  rmse "
              f"{np.sqrt(np.mean((pred-exact)**2)):.3e}  t=0 {e[m0].max():.3e}")

    # --- diffusion, Eq. (9.diffusion) ---
    def r_pde(P, X): return u_t(P, X) - u_xx(P, X)
    def r_ic(P, X):  return u_net(P, X) - np.sin(np.pi * X[:, 0])
    def r_bc(P, X):  return u_net(P, X)
    exact_d = np.exp(-np.pi**2 * X_eval[:, 1]) * np.sin(np.pi * X_eval[:, 0])

    for w, ni in [(1.0, 800), (10.0, 800), (100.0, 800), (10.0, 4000)]:
        terms = [("pde", 1.0, r_pde, X_col), ("ic", w, r_ic, X_ic),
                 ("bcL", w, r_bc, X_l), ("bcR", w, r_bc, X_r)]
        P, _ = pinn_solve(terms, [2, 30, 30, 1], "tanh", n_iter=ni, eta=1e-2,
                          rng=np.random.default_rng(1))
        report(u_net(P, X_eval), exact_d, f"diffusion soft, lam={w:g}, {ni} it")

    # --- wave, Eq. (9.wave); note the initial-velocity residual r_iv ---
    def w_pde(P, X): return u_tt(P, X) - u_xx(P, X)
    def r_iv(P, X):  return u_t(P, X)
    exact_w = np.cos(np.pi * X_eval[:, 1]) * np.sin(np.pi * X_eval[:, 0])

    for w, ni in [(10.0, 800), (10.0, 4000), (50.0, 4000)]:
        terms = [("pde", 1.0, w_pde, X_col), ("ic", w, r_ic, X_ic),
                 ("iv", w, r_iv, X_ic),
                 ("bcL", w, r_bc, X_l), ("bcR", w, r_bc, X_r)]
        P, _ = pinn_solve(terms, [2, 30, 30, 1], "tanh", n_iter=ni, eta=1e-2,
                          rng=np.random.default_rng(1))
        report(u_net(P, X_eval), exact_w, f"wave soft, lam={w:g}, {ni} it")
