"""Black-Scholes by a physics-informed network, Chapter 9."""
import autograd.numpy as np
from scipy.stats import norm
from nn_de import network, d_dxk, init_parameters
from pinn import pinn_solve

K, T, r, sigma, S_max = 5.0, 1.0, 0.05, 0.3, 20.0

def bs_exact(S, t):
    tau = np.maximum(T - t, 0.0)
    C = np.zeros_like(S)
    m = (tau > 0) & (S > 0)
    d1 = (np.log(S[m]/K) + (r + 0.5*sigma**2)*tau[m]) / (sigma*np.sqrt(tau[m]))
    d2 = d1 - sigma*np.sqrt(tau[m])
    C[m] = S[m]*norm.cdf(d1) - K*np.exp(-r*tau[m])*norm.cdf(d2)
    C[~m] = np.maximum(S[~m] - K, 0.0)
    return C

# ---- unscaled formulation: inputs (S, t) directly -------------------------
def make(scaled):
    """scaled=True works in s = S/K, c = C/K, tau = T - t."""
    Sc = K if scaled else 1.0
    Tmax = 1.0
    hi = S_max / Sc
    def C_net(P, X): return network(P, X, "tanh")
    C_t = d_dxk(C_net, 1); C_S = d_dxk(C_net, 0); C_SS = d_dxk(C_S, 0)
    def r_pde(P, X):
        S = X[:, 0]
        # in scaled variables tau = T - t so the sign of the time term flips
        return (-C_t(P, X) if scaled else C_t(P, X)) \
               + 0.5*sigma**2*S**2*C_SS(P, X) + r*S*C_S(P, X) - r*C_net(P, X)
    def r_term(P, X):                       # payoff at tau = 0 (t = T)
        return C_net(P, X) - np.maximum(X[:, 0] - K/Sc, 0.0)
    def r_lo(P, X):  return C_net(P, X)     # C(0, t) = 0
    def r_hi(P, X):
        tau = X[:, 1] if scaled else T - X[:, 1]
        return C_net(P, X) - (hi - (K/Sc)*np.exp(-r*tau))
    return C_net, r_pde, r_term, r_lo, r_hi, hi

def grids(hi, ns=40, nt=40, scaled=True):
    S = np.linspace(0, hi, ns); t = np.linspace(0, T, nt)
    Si, Ti = np.meshgrid(S[1:-1], t[1:-1], indexing="ij")
    X_col = np.column_stack([Si.ravel(), Ti.ravel()])
    X_term = np.column_stack([S, np.zeros(ns)])          # tau=0 (scaled) or t=0
    X_lo = np.column_stack([np.zeros(nt), t])
    X_hi = np.column_stack([hi*np.ones(nt), t])
    return X_col, X_term, X_lo, X_hi

def evaluate(C_net, P, scaled, n=120):
    Sc = K if scaled else 1.0
    S = np.linspace(0, S_max, n); t = np.linspace(0, T, n)
    Sg, Tg = np.meshgrid(S, t, indexing="ij")
    Xn = np.column_stack([(Sg/Sc).ravel(), ((T-Tg) if scaled else Tg).ravel()])
    pred = C_net(P, Xn)*Sc
    ex = bs_exact(Sg.ravel(), Tg.ravel())
    return pred, ex, Sg, Tg




# ---------------------------------------------------------------------------
# Reproduces Table 9.bsscaling, the price and Delta tables of Section 9.bsresults
# and the comparison against the closed form, Eq. (9.bsexact).
# ---------------------------------------------------------------------------
def main(n_iter=4000):
    import time
    from nn_de import network as _net
    results = {}
    for scaled in [False, True]:
        C_net, r_pde, r_term, r_lo, r_hi, hi = make(scaled)
        X_col, X_term, X_lo, X_hi = grids(hi, ns=30, nt=30, scaled=scaled)
        terms = [("pde", 1.0, r_pde, X_col), ("term", 10.0, r_term, X_term),
                 ("lo", 10.0, r_lo, X_lo),   ("hi", 10.0, r_hi, X_hi)]
        t0 = time.time()
        P, h = pinn_solve(terms, [2, 40, 40, 1], "tanh", n_iter=n_iter, eta=5e-3,
                          rng=np.random.default_rng(1), every=n_iter)
        pred, ex, Sg, Tg = evaluate(C_net, P, scaled)
        e = np.abs(pred - ex)
        tag = "scaled (s, tau)" if scaled else "unscaled (S, t)"
        print(f"{tag:18s} {time.time()-t0:5.1f}s  max {e.max():.4f}  "
              f"RMSE {np.sqrt(np.mean(e**2)):.4f}  L_pde {h[-1][2]['pde']:.2e}")
        results[scaled] = (P, pred, ex, Sg, Tg)

    P = results[True][0]
    pred, ex, Sg, Tg = results[True][1:]
    e = np.abs(pred - ex); i = e.argmax()
    print(f"\nmax err {e.max():.4f} at S={Sg.ravel()[i]:.2f} t={Tg.ravel()[i]:.3f} "
          f"(K={K}, T={T})   <- the kink of the payoff")

    print("\n--- price today (t=0) ---\n   S   network    exact   abs err")
    for S0 in [2., 4., 5., 6., 8., 12.]:
        X = np.array([[S0/K, T]])                     # tau = T at t = 0
        p = float(network(P, X, "tanh")[0] * K)
        x = float(bs_exact(np.array([S0]), np.array([0.0]))[0])
        print(f"{S0:5.1f}  {p:8.4f} {x:8.4f}   {abs(p-x):.4f}")

    dC = d_dxk(lambda P, X: network(P, X, "tanh"), 0)  # dC/dS = dc/ds
    print("\n--- Delta = dC/dS at t=0, exact = N(d1) ---\n   S   network    exact   abs err")
    for S0 in [2., 4., 5., 6., 8., 12.]:
        X = np.array([[S0/K, T]])
        dn = float(dC(P, X)[0])
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        print(f"{S0:5.1f}  {dn:8.4f} {norm.cdf(d1):8.4f}   {abs(dn-norm.cdf(d1)):.4f}")
    return P
