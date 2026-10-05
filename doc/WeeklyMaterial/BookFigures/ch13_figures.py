import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter13_transformers"))
from common import plt, np, save
import autograd.numpy as anp
from autograd import grad
import attention as at

fig,axs=plt.subplots(1,3,figsize=(13,3.8))

# (a) logit variance vs d_k
dks=np.array([4,8,16,32,64,128,256,512,1024])
vu,vs=[],[]
for dk in dks:
    r=np.random.default_rng(2)
    q=r.normal(size=(4000,dk)); k=r.normal(size=(4000,dk))
    s=np.sum(q*k,axis=1); vu.append(s.var()); vs.append((s/np.sqrt(dk)).var())
axs[0].loglog(dks,vu,"C0o-",label=r"unscaled $q\cdot k$")
axs[0].loglog(dks,dks,"k--",lw=1,label=r"prediction $d_k$")
axs[0].loglog(dks,vs,"C1s-",label=r"scaled by $1/\sqrt{d_k}$")
axs[0].axhline(1,color="0.6",lw=0.8)
axs[0].set_xlabel(r"$d_k$"); axs[0].set_ylabel("variance of the logits")
axs[0].set_title(r"(a) why the $1/\sqrt{d_k}$"); axs[0].legend(fontsize=8)

# (b) softmax saturation
scales=np.logspace(-1,2,25)
norms=[]
for c in scales:
    r=np.random.default_rng(3); s=r.normal(size=(1,32))*c
    f=lambda s: at.softmax_rows(s)[0,0]
    norms.append(np.linalg.norm(grad(f)(s)))
axs[1].loglog(scales,np.maximum(norms,1e-40),"C0o-",ms=3)
axs[1].axvline(1.0,color="k",ls="--",lw=0.9)
axs[1].set_xlabel("scale of the logits"); axs[1].set_ylabel(r"$\|\partial\,\mathrm{softmax}/\partial s\|_F$")
axs[1].set_title("(b) the softmax saturates")
axs[1].text(0.05,0.15,"unscaled logits\nland here for\nlarge $d_k$",transform=axs[1].transAxes,fontsize=8)

# (c) attention matrix of a trained two-block model would go here; instead show
# the attention pattern of a random model against a distance-decaying one
n=24
r=np.random.default_rng(0); d=16
P=at.init_mha(d,1,rng=r); X=r.normal(size=(n,d))
_,A=at.multihead(P,X)
im=axs[2].imshow(A[0],cmap="viridis")
plt.colorbar(im,ax=axs[2]); axs[2].grid(False)
axs[2].set_xlabel("key position $j$"); axs[2].set_ylabel("query position $i$")
axs[2].set_title(r"(c) an attention matrix $A_{ij}$")
save(fig,13,"attention_scaling")

# ---------------- figure 2: masks and positional encodings ----------------
fig,axs=plt.subplots(1,3,figsize=(13,3.6))
M=at.causal_mask(12)
_,Am=at.multihead(at.init_mha(16,1,rng=np.random.default_rng(0)),
                  np.random.default_rng(0).normal(size=(12,16)),M)
im=axs[0].imshow(Am[0],cmap="viridis"); plt.colorbar(im,ax=axs[0]); axs[0].grid(False)
axs[0].set_title("(a) causal mask: strictly lower triangular")
axs[0].set_xlabel("$j$"); axs[0].set_ylabel("$i$")

PE=at.positional_encoding(64,32)
im=axs[1].imshow(PE.T,aspect="auto",cmap="RdBu_r"); plt.colorbar(im,ax=axs[1])
axs[1].grid(False); axs[1].set_xlabel("position $m$"); axs[1].set_ylabel("channel $r$")
axs[1].set_title("(b) sinusoidal positional encoding")

G=PE@PE.T
axs[2].plot(G[32]/G[32].max(),"C0",lw=1.4)
axs[2].set_xlabel("position $m$"); axs[2].set_ylabel(r"$\mathrm{PE}(32)\cdot\mathrm{PE}(m)$, scaled")
axs[2].set_title("(c) the encoding is a similarity kernel")
axs[2].axvline(32,color="k",ls="--",lw=0.8)
save(fig,13,"attention_masks")

# --- appended: the framework experiments of Section 13.transflibraries ------
# Needs cross_check_attn.py, transformer_torch.py and transformer_tf.py to have
# been run in BookPrograms/chapter13_transformers.
RUN = os.environ.get("CH13_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter13_transformers"))
L = lambda n: np.load(os.path.join(RUN, n))

# d_k, var, ent_sc, ent_un, max_sc, max_un, jac_sc, jac_un
sc = L("scaling_table.npy")

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) entropy of an attention row
axs[0].semilogx(sc[:, 0], sc[:, 2], "C0o-", ms=5,
                label=r"scaled by $1/\sqrt{d_k}$")
axs[0].semilogx(sc[:, 0], sc[:, 3], "C3s--", ms=5, label="unscaled")
axs[0].axhline(np.log(64), color="k", ls=":", lw=1.0)
axs[0].annotate(r"$\log 64$", (sc[0, 0] * 1.2, np.log(64)),
                textcoords="offset points", xytext=(0, 4), fontsize=8)
axs[0].set_xlabel("$d_k$")
axs[0].set_ylabel("entropy of an attention row")
axs[0].set_title("(a) the rows collapse without it")
axs[0].legend(fontsize=8)
axs[0].set_xticks(sc[:, 0])
axs[0].set_xticklabels([int(v) for v in sc[:, 0]])

# (b) largest weight and softmax Jacobian
axs[1].semilogx(sc[:, 0], sc[:, 4], "C0o-", ms=5, label="max weight, scaled")
axs[1].semilogx(sc[:, 0], sc[:, 5], "C3s--", ms=5, label="max weight, unscaled")
axs[1].semilogx(sc[:, 0], sc[:, 6], "C0^-", ms=5, alpha=0.6,
                label=r"$\|\partial A/\partial S\|$, scaled")
axs[1].semilogx(sc[:, 0], sc[:, 7], "C3v--", ms=5, alpha=0.6,
                label=r"$\|\partial A/\partial S\|$, unscaled")
axs[1].set_xlabel("$d_k$")
axs[1].set_title("(b) and the derivative with them")
axs[1].legend(fontsize=7)
axs[1].set_xticks(sc[:, 0])
axs[1].set_xticklabels([int(v) for v in sc[:, 0]])

# (c) associative recall
tab, lens = L("recall_table.npy"), L("recall_lengths.npy")
names = ["transformer, 1 block", "transformer, 2 blocks",
         "2 blocks, learned pos.", "LSTM", "simple RNN"]
styles = ["C3s--", "C0o-", "C2^:", "C1d-.", "C4v--"]
for row, nm, st in zip(tab, names, styles):
    axs[2].plot(lens, row, st, ms=5, label=nm)
axs[2].axhline(0.125, color="k", ls=":", lw=1.0)
axs[2].annotate("chance", (lens[-1] * 0.6, 0.125), textcoords="offset points",
                xytext=(0, 5), fontsize=8)
axs[2].set_xlabel("key-value pairs $L$")
axs[2].set_ylabel("test accuracy")
axs[2].set_title("(c) associative recall")
axs[2].legend(fontsize=7, loc="upper right")
axs[2].set_xticks(lens)
axs[2].set_ylim(0, 1.08)

# (d) our gradients against PyTorch
errs = L("block_grad_errors.npy")
names_d = ["$W_Q$", "$W_K$", "$W_V$", "$W_O$", "$W_1$", "$b_1$", "$W_2$",
           "$b_2$", r"$\gamma_1$", r"$\beta_1$", r"$\gamma_2$", r"$\beta_2$"]
axs[3].bar(np.arange(len(errs)), errs, 0.6, color="C0")
axs[3].axhline(np.finfo(float).eps, color="k", ls=":", lw=1.0)
axs[3].annotate(r"$\epsilon_{\rm mach}$", (len(errs) - 2.5, np.finfo(float).eps),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[3].set_yscale("log")
axs[3].set_xticks(np.arange(len(errs)))
axs[3].set_xticklabels(names_d, fontsize=7, rotation=45)
axs[3].set_ylabel("max gradient difference")
axs[3].set_title("(d) a whole block, differentiated twice")
axs[3].set_ylim(1e-18, 1e-12)
save(fig, 13, "attention_frameworks")
