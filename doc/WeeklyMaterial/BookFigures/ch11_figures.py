import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter11_recurrent_networks"))
from common import plt, np, save
import rnn

T, n_h = 60, 20
def jac(rho, xscale=1e-3, seed=3):
    r=np.random.default_rng(seed)
    W=r.normal(0,1,(n_h,n_h)); W*= rho/np.max(np.abs(np.linalg.eigvals(W)))
    U=r.normal(0,1,(n_h,1)); X=r.normal(size=(T,1))*xscale
    h=np.zeros(n_h); Hs=[]
    for t in range(T):
        h=np.tanh(U@X[t]+W@h); Hs.append(h.copy())
    J=np.eye(n_h); out=[]
    for t in reversed(range(T)):
        J=J@(np.diag(1-Hs[t]**2)@W); out.append(np.linalg.norm(J,2))
    return np.array(out[::-1])

# ---------------- Figure 1: vanishing and exploding gradients --------------
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
lag=np.arange(T-1,-1,-1)
for rho,col in [(0.5,"C0"),(0.9,"C1"),(1.1,"C2"),(1.5,"C3")]:
    axs[0].semilogy(lag[::-1],jac(rho)[::-1],col,lw=1.6,label=rf"$\rho(W)={rho}$")
axs[0].axhline(1,color="k",ls=":",lw=0.8)
axs[0].set_xlabel(r"lag $T-t$"); axs[0].set_ylabel(r"$\|\partial h_T/\partial h_t\|_2$")
axs[0].set_title("(a) the product of Jacobians"); axs[0].legend(fontsize=8)

X=np.random.default_rng(3).normal(size=(T,1))*1e-3
for fb,col in [(0.0,"C0"),(1.0,"C1"),(2.0,"C2")]:
    P=rnn.init_lstm(1,n_h,1,np.random.default_rng(3),forget_bias=fb)
    _,F,_=rnn.lstm_forward(P,X)
    J=np.ones(n_h); out=[]
    for t in reversed(range(T)):
        J=J*F[t]; out.append(np.linalg.norm(J))
    axs[1].semilogy(lag[::-1],np.array(out[::-1])[::-1],col,lw=1.6,
                    label=rf"LSTM, $b_f={fb}$")
axs[1].semilogy(lag[::-1],jac(0.5)[::-1],"k--",lw=1.2,label=r"RNN, $\rho(W)=0.5$")
axs[1].set_xlabel(r"lag $T-t$"); axs[1].set_ylabel(r"$\|\partial c_T/\partial c_t\|$")
axs[1].set_title("(b) the LSTM cell-state path"); axs[1].legend(fontsize=8)

# (c) clipping
r=np.random.default_rng(0)
norms=np.abs(r.standard_cauchy(400))*0.5+0.2
theta=2.0
axs[2].semilogy(norms,"C0.",ms=3,label="raw gradient norm")
axs[2].semilogy(np.minimum(norms,theta),"C1.",ms=3,label=rf"clipped, $\theta={theta}$")
axs[2].axhline(theta,color="k",ls="--",lw=0.9)
axs[2].set_xlabel("update"); axs[2].set_ylabel(r"$\|\nabla\mathcal{L}\|$")
axs[2].set_title("(c) gradient clipping, Eq. (11.clip)"); axs[2].legend(fontsize=8)
save(fig,11,"rnn_gradients")

# ---------------- Figure 2: the damped oscillator -------------------------
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
S=np.load("osc_S.npy"); ts=np.load("osc_ts.npy")
axs[0].plot(ts,S[:,0],"C0",lw=1.4,label="$x(t)$")
axs[0].plot(ts,S[:,1],"C1",lw=1.0,label="$v(t)$")
axs[0].set_xlabel("$t$"); axs[0].set_ylabel("state")
axs[0].set_title(r"(a) $\ddot{x}+\eta\dot{x}+x=0$ by RK4"); axs[0].legend(fontsize=8)

h=np.load("osc_hist.npy")
axs[1].semilogy(h[:,0],h[:,1],"C0",lw=1.4)
axs[1].set_xlabel("epoch"); axs[1].set_ylabel("training cost")
axs[1].set_title("(b) BPTT with clipping, $\\theta=1$")

yp=np.load("osc_pred.npy").ravel(); yt=np.load("osc_true.npy").ravel()
axs[2].plot(yt,"k",lw=1.6,label="RK4, unseen initial condition")
axs[2].plot(yp,"C3--",lw=1.2,label="RNN one-step prediction")
axs[2].set_xlabel("step within the test sequences"); axs[2].set_ylabel("$x$")
axs[2].set_title("(c) generalisation to a new trajectory"); axs[2].legend(fontsize=8)
save(fig,11,"rnn_oscillator")

# --- appended: the framework experiments of Section 11.rnnlibraries ---------
# Needs cross_check_rnn.py, rnn_torch.py and rnn_tf.py to have been run in
# BookPrograms/chapter11_recurrent_networks.
RUN = os.environ.get("CH11_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter11_recurrent_networks"))
L = lambda n: np.load(os.path.join(RUN, n))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) the sine forecast
true, pt = L("sine_true.npy"), L("sine_pred.npy")
pk = L("sine_pred_keras.npy")
axs[0].plot(true, "k-", lw=2.2, label="true")
axs[0].plot(pt, "C0--", lw=1.4, label="PyTorch nn.RNN")
axs[0].plot(pk, "C1:", lw=1.6, label="Keras SimpleRNN")
axs[0].set_xlabel("test step")
axs[0].set_ylabel("$x$")
axs[0].set_title("(a) one step ahead")
axs[0].legend(fontsize=8)

# (b) MNIST row by row
for nm, f, c, ls in [("LSTM, PyTorch", "mnist_lstm.npy", "C0", "-"),
                     ("LSTM, Keras", "mnist_lstm_keras.npy", "C1", "--"),
                     ("RNN, PyTorch", "mnist_rnn.npy", "C2", "-"),
                     ("RNN, Keras", "mnist_rnn_keras.npy", "C3", "--")]:
    h = L(f)
    axs[1].plot(h[:, 0], h[:, 2], ls, color=c, marker="o", ms=4, label=nm)
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("test accuracy")
axs[1].set_title("(b) MNIST as 28 rows of 28")
axs[1].legend(fontsize=7.5, loc="lower right")
axs[1].set_xticks([1, 2, 3])

# (c) the adding problem
tab, lags, base = L("adding_table.npy"), L("adding_lags.npy"), L("adding_baseline.npy")
for i, (nm, c) in enumerate([("simple RNN", "C3"), ("LSTM", "C0"), ("GRU", "C2")]):
    axs[2].semilogy(lags, np.maximum(tab[i], 1e-6), "o-", color=c, label=nm)
axs[2].semilogy(lags, base, "k:", lw=1.4, label="predict the mean")
axs[2].set_xlabel("sequence length $T$")
axs[2].set_ylabel("test mean squared error")
axs[2].set_title("(c) the adding problem")
axs[2].legend(fontsize=8)

# (d) our BPTT against autograd
names = ["$U$", "$W$", "$V$", "$b$", "$c$"]
errs = L("bptt_errors.npy")
axs[3].bar(np.arange(len(names)), errs, 0.55, color="C0")
axs[3].axhline(np.finfo(float).eps, color="k", ls=":", lw=1.0)
axs[3].annotate(r"$\epsilon_{\rm mach}$", (len(names) - 1.4, np.finfo(float).eps),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[3].set_yscale("log")
axs[3].set_xticks(np.arange(len(names)))
axs[3].set_xticklabels(names)
axs[3].set_ylabel("max gradient difference")
axs[3].set_title("(d) our BPTT vs autograd")
axs[3].set_ylim(1e-18, 1e-12)
save(fig, 11, "rnn_frameworks")
