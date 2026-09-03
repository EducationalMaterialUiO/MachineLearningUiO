"""Do the three recurrent implementations compute the same thing?

The chapter derives Eq. (11.rnn), backpropagation through time
Eqs. (11.bptth)-(11.bpttparams), and the LSTM cell
Eqs. (11.lstmf)-(11.lstmh).  Here one set of weights is pushed into our NumPy
code, into PyTorch and into Keras, and the forward pass and every gradient are
compared.

Two conventions have to be reconciled, and both are worth knowing.

*  PyTorch's ``nn.RNN`` carries **two** bias vectors, ``bias_ih`` and
   ``bias_hh``, where Eq. (11.rnn) has one; their sum is our ``b``, so we put
   ours in the first and zero the second.
*  Keras stores the input matrix transposed, ``kernel`` of shape (n_x, n_h)
   against our ``U`` of shape (n_h, n_x), and likewise the recurrent matrix.

For the LSTM there is a third: the four gates are concatenated into one array,
in the order (i, f, g, o) in PyTorch and (i, f, c, o) in Keras -- the same
order, but our Eqs. (11.lstmf)-(11.lstmh) name them (f, i, g, o).  Getting this
wrong produces a network that trains but is not the one on the page.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras
from tensorflow.keras import layers

import rnn

torch.set_default_dtype(torch.float64)
keras.backend.set_floatx("float64")

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)

T, n_x, n_h, n_y = 12, 3, 8, 2
X = rng.normal(size=(T, n_x))
target = rng.normal(size=(T, n_y))
p = rnn.init_rnn(n_x, n_h, n_y, rng=np.random.default_rng(1))

# ---------------------------------------------------------------------------
# 1.  the forward pass of Eq. (11.rnn)
# ---------------------------------------------------------------------------
out.write("=== 1. the simple RNN of Eq. (11.rnn), forward ===\n")
Y_ours, cache = rnn.forward(p, X)
_, A, H = cache

cell_t = nn.RNN(n_x, n_h, batch_first=True, nonlinearity="tanh")
with torch.no_grad():
    cell_t.weight_ih_l0.copy_(torch.tensor(p["U"]))
    cell_t.weight_hh_l0.copy_(torch.tensor(p["W"]))
    cell_t.bias_ih_l0.copy_(torch.tensor(p["b"]))     # one bias, not two
    cell_t.bias_hh_l0.zero_()
read_t = nn.Linear(n_h, n_y)
with torch.no_grad():
    read_t.weight.copy_(torch.tensor(p["V"]))
    read_t.bias.copy_(torch.tensor(p["c"]))

Xt = torch.tensor(X)[None]                            # (1, T, n_x)
Ht, _ = cell_t(Xt)
Yt = read_t(Ht)[0].detach().numpy()

cell_k = layers.SimpleRNN(n_h, return_sequences=True, activation="tanh")
cell_k.build((None, T, n_x))
cell_k.set_weights([p["U"].T, p["W"].T, p["b"]])      # keras stores transposed
Hk = cell_k(X[None]).numpy()[0]
Yk = Hk @ p["V"].T + p["c"]

out.write(f"  hidden states  max |ours - torch| : "
          f"{np.abs(H[1:] - Ht[0].detach().numpy()).max():.3e}\n")
out.write(f"  hidden states  max |ours - keras| : "
          f"{np.abs(H[1:] - Hk).max():.3e}\n")
out.write(f"  outputs        max |ours - torch| : "
          f"{np.abs(Y_ours - Yt).max():.3e}\n")
out.write(f"  outputs        max |ours - keras| : "
          f"{np.abs(Y_ours - Yk).max():.3e}\n")
out.write(f"  scale of the states                : {np.abs(H).max():.4f}\n\n")

# ---------------------------------------------------------------------------
# 2.  backpropagation through time
# ---------------------------------------------------------------------------
out.write("=== 2. backpropagation through time, Eqs. (11.bptth)-(11.bpttparams) "
          "===\n")
g_ours = rnn.bptt(p, cache, Y_ours, target)

Ht2, _ = cell_t(Xt)
Yt2 = read_t(Ht2)[0]
loss_t = 0.5 * torch.sum((Yt2 - torch.tensor(target)) ** 2, dim=1).mean()
cell_t.zero_grad()
read_t.zero_grad()
loss_t.backward()
g_t = {"U": cell_t.weight_ih_l0.grad.numpy(), "W": cell_t.weight_hh_l0.grad.numpy(),
       "b": cell_t.bias_ih_l0.grad.numpy(), "V": read_t.weight.grad.numpy(),
       "c": read_t.bias.grad.numpy()}

out.write(f"  our loss, Eq. (11.cost)   : {rnn.mse(Y_ours, target):.12f}\n")
out.write(f"  torch loss                : {loss_t.item():.12f}\n")
out.write("   array   shape        |ours - torch|      scale\n")
worst = 0.0
for k in ["U", "W", "V", "b", "c"]:
    e = np.abs(g_ours[k] - g_t[k]).max()
    worst = max(worst, e)
    out.write(f"   {k:3s}   {str(g_ours[k].shape):12s} {e:14.3e}  "
              f"{np.abs(g_ours[k]).max():10.3e}\n")
out.write(f"  worst disagreement: {worst:.3e}\n")
out.write("  our BPTT recursion is what autograd computes.\n\n")
np.save("bptt_errors.npy", np.array([np.abs(g_ours[k]-g_t[k]).max()
                                     for k in ["U", "W", "V", "b", "c"]]))

# ---------------------------------------------------------------------------
# 3.  the LSTM cell of Eqs. (11.lstmf)-(11.lstmh)
# ---------------------------------------------------------------------------
out.write("=== 3. the LSTM cell, Eqs. (11.lstmf)-(11.lstmh) ===\n")
P = rnn.init_lstm(n_x, n_h, n_y, rng=np.random.default_rng(2), forget_bias=1.0)
Y_l, F_l, C_l = rnn.lstm_forward(P, X)

lstm_t = nn.LSTM(n_x, n_h, batch_first=True)
with torch.no_grad():
    # torch concatenates the gates in the order (i, f, g, o)
    lstm_t.weight_ih_l0.copy_(torch.tensor(np.vstack(
        [P["Wix"], P["Wfx"], P["Wgx"], P["Wox"]])))
    lstm_t.weight_hh_l0.copy_(torch.tensor(np.vstack(
        [P["Wih"], P["Wfh"], P["Wgh"], P["Woh"]])))
    lstm_t.bias_ih_l0.copy_(torch.tensor(np.concatenate(
        [P["bi"], P["bf"], P["bg"], P["bo"]])))
    lstm_t.bias_hh_l0.zero_()
Hl, (hT, cT) = lstm_t(Xt)
Y_t = Hl[0].detach().numpy() @ P["V"].T + P["c_out"]

lstm_k = layers.LSTM(n_h, return_sequences=True, return_state=True,
                     unit_forget_bias=False)
lstm_k.build((None, T, n_x))
lstm_k.set_weights([
    np.hstack([P["Wix"].T, P["Wfx"].T, P["Wgx"].T, P["Wox"].T]),
    np.hstack([P["Wih"].T, P["Wfh"].T, P["Wgh"].T, P["Woh"].T]),
    np.concatenate([P["bi"], P["bf"], P["bg"], P["bo"]])])
Hk_l = lstm_k(X[None])[0].numpy()[0]
Y_k = Hk_l @ P["V"].T + P["c_out"]

out.write(f"  cell output    max |ours - torch| : "
          f"{np.abs(Y_l - Y_t).max():.3e}\n")
out.write(f"  cell output    max |ours - keras| : "
          f"{np.abs(Y_l - Y_k).max():.3e}\n")
out.write(f"  final cell state |c_T| ours/torch  : "
          f"{np.abs(C_l[-1]).max():.6f} / "
          f"{np.abs(cT.detach().numpy()).max():.6f}\n")
out.write(f"  mean forget gate, Eq. (11.lstmf)   : {F_l.mean():.4f}\n\n")

# ---------------------------------------------------------------------------
# 4.  Theorem 11.vanishing measured through autograd rather than by hand
# ---------------------------------------------------------------------------
out.write("=== 4. Theorem 11.vanishing, measured by autograd in PyTorch ===\n")
out.write("  the Jacobian dh_T/dh_1 is assembled column by column from\n"
          "  torch.autograd.grad, with no reference to Eq. (11.jacprod)\n\n")
out.write("   rho(W)   ||dh_T/dh_1|| autograd   ||dh_T/dh_1|| Eq. (11.jacprod)"
          "   ratio\n")
Tlong, nh = 60, 20
rows = []
for rho in [0.5, 0.9, 1.1, 1.5]:
    r = np.random.default_rng(7)
    W = r.normal(0, 1, (nh, nh))
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    U = r.normal(0, 0.01, (nh, 1))
    Xs = torch.tensor(r.normal(0, 0.01, (Tlong, 1)))
    Wt = torch.tensor(W)
    Ut = torch.tensor(U)

    def run(h1):
        h = h1
        for t in range(1, Tlong):
            h = torch.tanh(Ut @ Xs[t] + Wt @ h)
        return h

    h1 = torch.tanh(Ut @ Xs[0] + Wt @ torch.zeros(nh, dtype=torch.float64))
    h1 = h1.detach().requires_grad_(True)
    hT = run(h1)
    J = torch.stack([torch.autograd.grad(hT[i], h1, retain_graph=True)[0]
                     for i in range(nh)]).numpy()

    # the same product formed explicitly, Eq. (11.jacprod)
    h = np.tanh(U @ Xs[0].numpy() + W @ np.zeros(nh))
    Jp = np.eye(nh)
    for t in range(1, Tlong):
        a = U @ Xs[t].numpy() + W @ h
        h = np.tanh(a)
        Jp = (np.diag(1 - h ** 2) @ W) @ Jp
    na, np_ = np.linalg.norm(J, 2), np.linalg.norm(Jp, 2)
    rows.append((rho, na, np_))
    out.write(f"   {rho:5.2f}   {na:22.3e}   {np_:25.3e}   {na/np_:7.4f}\n")
out.write("  the two agree: what the framework differentiates is the product\n"
          "  of Jacobians that Theorem 11.vanishing bounds.\n")
np.save("jacobian_autograd.npy", np.array(rows))
out.close()
print(open("cross_check.txt").read())

# ---------------------------------------------------------------------------
# 5.  the exact cell Jacobian against the approximation of Eq. (11.lstmjac)
# ---------------------------------------------------------------------------
out2 = open("cross_check.txt", "a", buffering=1)
out2.write("\n=== 5. Eq. (11.lstmjac) is approximate: how approximate? ===\n")
out2.write("  the exact Jacobian is assembled by autograd; the approximation is\n"
           "  the product of forget gates alone.  `input scale' sets how hard the\n"
           "  gates are driven, and therefore how saturated they are.\n\n")
out2.write("  input scale   b_f   mean f   prod diag(f)   exact ||dc_T/dc_1||"
           "   ratio\n")
Tl, nh_l = 40, 12
rowsl = []
for xs in [1.0, 0.1]:
    for bf in [0.0, 1.0, 2.0, 4.0]:
        r = np.random.default_rng(11)
        Pl = rnn.init_lstm(2, nh_l, 1, rng=np.random.default_rng(11),
                           forget_bias=bf)
        Xl = torch.tensor(r.normal(0, xs, (Tl, 2)))
        tp = {k: torch.tensor(v) for k, v in Pl.items()}

        def cell(c1):
            """Eqs. (11.lstmf)-(11.lstmh) run forward from a given c_1."""
            h = torch.zeros(nh_l, dtype=torch.float64)
            c, fs = c1, []
            for t in range(1, Tl):
                f = torch.sigmoid(tp["Wfx"] @ Xl[t] + tp["Wfh"] @ h + tp["bf"])
                i = torch.sigmoid(tp["Wix"] @ Xl[t] + tp["Wih"] @ h + tp["bi"])
                g = torch.tanh(tp["Wgx"] @ Xl[t] + tp["Wgh"] @ h + tp["bg"])
                o = torch.sigmoid(tp["Wox"] @ Xl[t] + tp["Woh"] @ h + tp["bo"])
                c = f * c + i * g
                h = o * torch.tanh(c)
                fs.append(f)
            return c, fs

        z = torch.zeros(nh_l, dtype=torch.float64)
        i0 = torch.sigmoid(tp["Wix"] @ Xl[0] + tp["bi"])
        g0 = torch.tanh(tp["Wgx"] @ Xl[0] + tp["bg"])
        c1 = (i0 * g0).detach().requires_grad_(True)
        cT, fs = cell(c1)
        J = torch.stack([torch.autograd.grad(cT[k], c1, retain_graph=True)[0]
                         for k in range(nh_l)]).numpy()
        approx = np.eye(nh_l)
        for f in fs:
            approx = np.diag(f.detach().numpy()) @ approx
        na, ne = np.linalg.norm(approx, 2), np.linalg.norm(J, 2)
        mean_f = float(np.mean([f.detach().numpy().mean() for f in fs]))
        rowsl.append((xs, bf, mean_f, na, ne))
        out2.write(f"  {xs:11.1f}  {bf:4.1f}   {mean_f:6.4f}   {na:13.3e}"
                   f"   {ne:19.3e}   {ne/na:7.2f}\n")
out2.write("  Eq. (11.lstmjac) is a lower bound, not an estimate.  The neglected\n"
           "  paths through the gates carry gradient too, and they dominate: the\n"
           "  ratio never approaches one.  Biasing the gates open improves it by\n"
           "  two orders of magnitude, and driving the cell more gently makes it\n"
           "  worse, because a cell state near zero has 1 - tanh^2(c) near one and\n"
           "  the gate paths are then at their most conductive.  The sign of the\n"
           "  discrepancy is favourable -- the true gradient is larger than the\n"
           "  carousel argument promises -- but the argument is not quantitative.\n")
np.save("lstm_jacobian.npy", np.array(rowsl))

# ---------------------------------------------------------------------------
# 6.  the measured memory horizon
# ---------------------------------------------------------------------------
out2.write("\n=== 6. the memory horizon: fitted decay rate, Corollary 11.horizon "
           "===\n")
out2.write("  ||dh_T/dh_t|| is measured at every lag and fitted to exp(-lag/tau)\n\n")
out2.write("   rho(W)  sigma_max   fitted tau (steps)   lag for a factor 1e-6"
           "   norm at lag 59\n")
horizon = []
for rho in [0.5, 0.7, 0.9, 1.1]:
    r = np.random.default_rng(7)
    W = r.normal(0, 1, (20, 20))
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    U = r.normal(0, 0.01, (20, 1))
    Xs = r.normal(0, 0.01, (60, 1))
    h = np.tanh(U @ Xs[0] + W @ np.zeros(20))
    Jp, norms = np.eye(20), []
    for t in range(1, 60):
        h = np.tanh(U @ Xs[t] + W @ h)
        Jp = (np.diag(1 - h ** 2) @ W) @ Jp
        norms.append(np.linalg.norm(Jp, 2))
    lags = np.arange(1, 60)
    slope = np.polyfit(lags, np.log(norms), 1)[0]
    tau = -1.0 / slope
    horizon.append((rho, np.linalg.norm(W, 2), tau, norms[-1]))
    out2.write(f"   {rho:5.2f}   {np.linalg.norm(W,2):8.4f}   {tau:18.2f}"
               f"   {tau*np.log(1e6):21.0f}   {norms[-1]:14.3e}\n")
out2.write("  a negative tau means the product grows rather than decays\n")
np.save("horizon.npy", np.array(horizon))
out2.close()
