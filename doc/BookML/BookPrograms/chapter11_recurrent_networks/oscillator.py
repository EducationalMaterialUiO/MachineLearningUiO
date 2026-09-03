"""The damped harmonic oscillator of Section 11.oscillator, integrated by RK4
and then learned by the recurrent network of rnn.py."""
import numpy as np, rnn

eta = 0.2                      # damping
def deriv(s, t, F=lambda t: 0.0):
    x, v = s
    return np.array([v, F(t) - eta*v - x])

def rk4(s0, ts, F=lambda t: 0.0):
    S = np.zeros((len(ts), 2)); S[0] = s0
    for i in range(len(ts)-1):
        h = ts[i+1]-ts[i]; s = S[i]; t = ts[i]
        k1 = deriv(s, t, F); k2 = deriv(s+0.5*h*k1, t+0.5*h, F)
        k3 = deriv(s+0.5*h*k2, t+0.5*h, F); k4 = deriv(s+h*k3, t+h, F)
        S[i+1] = s + h*(k1+2*k2+2*k3+k4)/6
    return S

if __name__ == "__main__":
    ts = np.linspace(0, 40, 800)
    L = 40

    def make(s0, F=lambda t: 0.0, stride=5):
        S = rk4(np.array(s0), ts, F)
        x, v = S[:, 0], S[:, 1]
        # input x_i, target the next value x_{i+1}: one-step-ahead forecasting
        seqs = [x[i:i+L, None] for i in range(0, len(x)-L-1, stride)]
        tgts = [x[i+1:i+L+1, None] for i in range(0, len(x)-L-1, stride)]
        return seqs, tgts, S

    # train on one trajectory, test on initial conditions never seen
    tr_s, tr_t, S_tr = make([1.0, 0.0])
    te1_s, te1_t, S_te1 = make([0.0, 1.0])                       # different IC
    te2_s, te2_t, S_te2 = make([0.5, -0.3],
                               F=lambda t: 0.3*np.cos(0.7*t))    # driven
    print(f"{len(tr_s)} training sequences of length {L}")
    res = {}
    for seed in range(3):
        p = rnn.init_rnn(1, 24, 1, np.random.default_rng(seed))
        p, hist = rnn.train(p, tr_s, tr_t, n_epoch=300, eta=5e-3, theta=1.0,
                            rng=np.random.default_rng(seed),
                            verbose=(seed == 0), every=100)
        e1 = np.mean([rnn.mse(rnn.forward(p, a)[0], b) for a, b in zip(te1_s, te1_t)])
        e2 = np.mean([rnn.mse(rnn.forward(p, a)[0], b) for a, b in zip(te2_s, te2_t)])
        res.setdefault("ic", []).append(e1); res.setdefault("driven", []).append(e2)
        if seed == 0:
            np.save("osc_hist.npy", np.array(hist))
            Yp = np.concatenate([rnn.forward(p, a)[0] for a in te1_s[:12]])
            Yt = np.concatenate(te1_t[:12])
            np.save("osc_pred.npy", Yp); np.save("osc_true.npy", Yt)
            np.save("osc_S.npy", S_tr); np.save("osc_ts.npy", ts)
    vt1 = S_te1[:, 0].var(); vt2 = S_te2[:, 0].var()
    a1, a2 = np.array(res["ic"]), np.array(res["driven"])
    print(f"unseen initial condition (0,1):   MSE {a1.mean():.3e}   "
          f"relative to Var(x)={vt1:.4f}: {a1.mean()/vt1:.2%}")
    print(f"driven, unseen forcing:           MSE {a2.mean():.3e}   "
          f"relative to Var(x)={vt2:.4f}: {a2.mean()/vt2:.2%}")
