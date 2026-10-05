import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter12_autoencoders"))
from common import plt, np, save
import ae

# ---------------- Figure 1: the linear autoencoder is PCA ------------------
lam=np.load("pca_lam.npy"); hist=np.load("pca_hist.npy")
X=np.load("pca_X.npy"); Up=np.load("pca_Up.npy"); p=3
fig,axs=plt.subplots(1,3,figsize=(13,3.8))

k=np.arange(1,len(lam)+1)
axs[0].bar(k[:p],lam[:p],color="C0",label=r"kept, $i\leq p$")
axs[0].bar(k[p:],np.maximum(lam[p:],1e-4),color="C3",label=r"discarded, $i>p$")
axs[0].set_yscale("log"); axs[0].set_xlabel(r"$i$"); axs[0].set_ylabel(r"$\lambda_i$")
axs[0].set_title("(a) spectrum of the covariance $\\mathbf{S}$")
axs[0].text(0.97,0.60,rf"$\sum_{{i>p}}\lambda_i={lam[p:].sum():.4f}$"+"\n= the error floor",
            transform=axs[0].transAxes,ha="right",fontsize=9,
            bbox=dict(fc="white",ec="0.7"))
axs[0].legend(fontsize=8,loc="upper right")

floor=float(np.load("pca_floor.npy")[0])
axs[1].semilogy(hist[:,0],hist[:,1]-floor,"C0",lw=1.2)
axs[1].set_xlabel("epoch")
axs[1].set_ylabel(r"AE error $-\;\sum_{i>p}\lambda_i$")
axs[1].set_title("(b) the autoencoder descends to the floor")
axs[1].text(0.97,0.9,"the floor is never crossed",transform=axs[1].transAxes,
            ha="right",va="top",fontsize=9)

sv=np.load("pca_sv.npy")
axs[2].semilogy(np.arange(1,len(sv)+1),np.maximum(sv,1e-18),"C0o",ms=7)
axs[2].axhline(1.0,color="k",ls="--",lw=0.9)
axs[2].set_xlabel(r"$i$"); axs[2].set_ylabel(r"$\sigma_i(W_eW_d)$")
axs[2].set_title(r"(c) $W_eW_d$ is a rank-$p$ projector")
axs[2].text(0.5,0.55,f"$p={p}$ singular values at 1,\n"+r"$d-p$ at $10^{-16}$",
            transform=axs[2].transAxes,ha="center",fontsize=9,
            bbox=dict(fc="white",ec="0.7"))
save(fig,12,"ae_pca")

# ---------------- Figure 2: beyond PCA -------------------------------------
Xn=np.load("nl_X.npy"); t=np.load("nl_t.npy")
rec=np.load("nl_rec.npy"); code=np.load("nl_code.npy").ravel()
pca1=np.load("nl_pca1.npy")
fig=plt.figure(figsize=(13,3.9))
ax=fig.add_subplot(1,3,1,projection="3d")
ax.scatter(Xn[:,0],Xn[:,1],Xn[:,2],c=t,cmap="viridis",s=6)
ax.scatter(pca1[:,0],pca1[:,1],pca1[:,2],c="C3",s=4,alpha=0.5)
ax.set_title("(a) data and the PCA line, $p=1$",fontsize=10)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
ax2=fig.add_subplot(1,3,2,projection="3d")
ax2.scatter(Xn[:,0],Xn[:,1],Xn[:,2],c=t,cmap="viridis",s=6,alpha=0.35)
ax2.scatter(rec[:,0],rec[:,1],rec[:,2],c="C3",s=4)
ax2.set_title("(b) the nonlinear autoencoder, $p=1$",fontsize=10)
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$"); ax2.set_zlabel("$x_3$")
ax3=fig.add_subplot(1,3,3)
ax3.scatter(t,code,c=t,cmap="viridis",s=6)
ax3.set_xlabel("true parameter $t$ along the curve"); ax3.set_ylabel("learned code $z$")
ax3.set_title("(c) the code is only piecewise monotone in $t$",fontsize=10)
plt.tight_layout()
save(fig,12,"ae_nonlinear")

# --- appended: the framework experiments of Section 12.aelibraries ----------
# Needs cross_check_ae.py, ae_torch.py and ae_tf.py to have been run in
# BookPrograms/chapter12_autoencoders.
RUN = os.environ.get("CH12_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter12_autoencoders"))
L = lambda n: np.load(os.path.join(RUN, n))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) reconstruction error against the code dimension
c = L("mnist_curve.npy")                    # p, pca, linear, nonlinear
axs[0].loglog(c[:, 0], c[:, 1], "ko-", ms=5, label="PCA (linear optimum)")
axs[0].loglog(c[:, 0], c[:, 2], "C0s--", ms=5, label="linear autoencoder")
axs[0].loglog(c[:, 0], c[:, 3], "C3^-", ms=5, label="nonlinear autoencoder")
axs[0].set_xlabel("code dimension $p$")
axs[0].set_ylabel("test mse per pixel")
axs[0].set_title("(a) what the bottleneck costs")
axs[0].legend(fontsize=8)
axs[0].set_xticks(c[:, 0])
axs[0].set_xticklabels([int(v) for v in c[:, 0]])

# (b) the principal angle falling to zero
ht = L("angle_hist_torch.npy")              # iter, loss, angle
hk = L("angle_hist_keras.npy")              # iter, loss, angle
axs[1].semilogy(ht[:, 0], ht[:, 2], "C0-", lw=1.6, label="PyTorch")
axs[1].semilogy(hk[:, 0], hk[:, 2], "C1--", lw=1.6, label="TensorFlow")
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("largest principal angle (degrees)")
axs[1].set_title("(b) Theorem 12.aepca being reached")
axs[1].legend(fontsize=9)

# (c) reconstructions at p = 16
orig, rec = L("orig8.npy"), L("recon16.npy")
grid = np.zeros((2 * 28, 8 * 28))
for k in range(8):
    grid[:28, k * 28:(k + 1) * 28] = orig[k].reshape(28, 28)
    grid[28:, k * 28:(k + 1) * 28] = rec[k].reshape(28, 28)
axs[2].imshow(grid, cmap="gray_r")
axs[2].set_xticks([])
axs[2].set_yticks([28 * 0.5, 28 * 1.5])
axs[2].set_yticklabels(["data", "$p=16$"], fontsize=9)
axs[2].grid(False)
axs[2].set_title("(c) reconstructions")

# (d) our gradients against the frameworks'
errs = L("ae_grad_errors.npy")
names = [f"layer {i}" for i in range(len(errs))]
axs[3].bar(np.arange(len(errs)), errs, 0.55, color="C0")
axs[3].axhline(np.finfo(float).eps, color="k", ls=":", lw=1.0)
axs[3].annotate(r"$\epsilon_{\rm mach}$", (len(errs) - 1.4, np.finfo(float).eps),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[3].set_yscale("log")
axs[3].set_xticks(np.arange(len(errs)))
axs[3].set_xticklabels(names, fontsize=8)
axs[3].set_ylabel("max gradient difference")
axs[3].set_title("(d) our gradients vs autograd")
axs[3].set_ylim(1e-18, 1e-12)
save(fig, 12, "ae_frameworks")
