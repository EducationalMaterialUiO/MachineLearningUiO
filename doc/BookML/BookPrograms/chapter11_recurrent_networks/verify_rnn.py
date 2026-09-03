"""Chapter 11: every verification quoted in Sections 11.6, 11.5.1 and 11.7.1.

Run with:  python3 verify_rnn.py
"""
import numpy as np
import rnn

# --- 1. BPTT against central differences, Section 11.rnncode ---------------
print("--- BPTT vs central differences (T=12) ---")
rng = np.random.default_rng(0)
p = rnn.init_rnn(2, 6, 1, rng)
T = 12
X = rng.normal(size=(T, 2)); tgt = rng.normal(size=(T, 1))
Y, cache = rnn.forward(p, X)
g = rnn.bptt(p, cache, Y, tgt)


def loss():
    Yl, _ = rnn.forward(p, X)
    return rnn.mse(Yl, tgt)


for k in ["U", "W", "V", "b", "c"]:
    A = p[k]; err = 0.0
    it = np.nditer(A, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; old = A[i]; h = 1e-6
        A[i] = old + h; fp = loss()
        A[i] = old - h; fm = loss()
        A[i] = old
        num = (fp - fm) / (2 * h)
        err = max(err, abs(num - g[k][i]) / max(1e-12, abs(num) + abs(g[k][i])))
        it.iternext()
    print(f"  {k}: max relative error over all {A.size} entries = {err:.2e}")

# --- 2. the Jacobian product, Theorem 11.vanishing -------------------------
T, n_h = 60, 20


def jac(rho, xscale=1e-3, seed=3):
    """||dh_T/dh_t|| for t = 1..T, Eq. (11.jacprod)."""
    r = np.random.default_rng(seed)
    W = r.normal(0, 1, (n_h, n_h))
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    U = r.normal(0, 1, (n_h, 1)); X = r.normal(size=(T, 1)) * xscale
    h = np.zeros(n_h); Hs = []
    for t in range(T):
        h = np.tanh(U @ X[t] + W @ h); Hs.append(h.copy())
    J = np.eye(n_h); out = []
    for t in reversed(range(T)):
        J = J @ (np.diag(1 - Hs[t] ** 2) @ W)
        out.append(np.linalg.norm(J, 2))
    return np.array(out[::-1]), np.linalg.norm(W, 2), \
        np.mean([np.max(1 - h ** 2) for h in Hs])


print("\n=== small inputs: the network stays in the linear regime, tanh' ~ 1 ===")
print(" rho(W)  sigma_max  mean max(tanh')   ||dh_T/dh_1||   bound sigma_max^59")
for rho in [0.5, 0.9, 1.1, 1.5]:
    n, s, d = jac(rho)
    print(f"  {rho:4.2f}  {s:8.3f}  {d:12.4f}   {n[0]:.3e}     {s**59:.3e}")

# --- 3. the LSTM cell path, Section 11.lstmwhy -----------------------------
print("\n=== LSTM cell-state path, Eq. (11.lstmjac) ===")
X = np.random.default_rng(3).normal(size=(T, 1)) * 1e-3
for fb in [0.0, 1.0, 2.0]:
    P = rnn.init_lstm(1, n_h, 1, np.random.default_rng(3), forget_bias=fb)
    _, F, _ = rnn.lstm_forward(P, X)
    J = np.ones(n_h)
    for t in reversed(range(T)):
        J = J * F[t]
    print(f"  forget bias {fb:.1f}: mean f={F.mean():.4f}  "
          f"||dc_T/dc_1||={np.linalg.norm(J):.3e}")
print(f"  (vanilla RNN at rho=0.5 for comparison: {jac(0.5)[0][0]:.3e})")
