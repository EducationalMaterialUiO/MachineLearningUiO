import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter15_vae"))
from common import plt, np, save
import autograd.numpy as anp
import vae
from sklearn.datasets import load_digits

fig,axs=plt.subplots(1,3,figsize=(13,3.8))

# (a) training curves for several latent dimensions
d=load_digits(); X=(d.images.reshape(len(d.images),-1)/16.0); X=(X>0.5).astype(float)
Xtr=X[:1200]
for dh,c in [(2,"C0"),(5,"C1"),(10,"C2"),(20,"C3")]:
    P=vae.init_vae(X.shape[1],dh,hidden=64,rng=anp.random.default_rng(0))
    P,h=vae.train_vae(P,Xtr,n_iter=1500,batch=64,gamma=2e-3,
                      rng=anp.random.default_rng(1),every=100)
    h=np.array(h)
    axs[0].plot(h[:,0],h[:,1],c,lw=1.4,label=rf"$d_h={dh}$")
axs[0].set_xlabel("iteration"); axs[0].set_ylabel("ELBO")
axs[0].set_title("(a) the ELBO rises"); axs[0].legend(fontsize=8,loc="lower right")

# (b) posterior collapse: KL carried by each latent unit
K=np.load("vae_klall.npy")
for row,dh,c in zip(K,[2,5,10,20],["C0","C1","C2","C3"]):
    n=dh; v=np.sort(row[:n])[::-1]
    axs[1].semilogy(np.arange(1,n+1),np.maximum(v,1e-6),c+"o-",ms=4,label=rf"$d_h={dh}$")
axs[1].axhline(0.01,color="k",ls="--",lw=1)
axs[1].text(11,0.013,"collapse threshold",fontsize=8)
axs[1].set_xlabel("latent unit, sorted"); axs[1].set_ylabel(r"$\mathrm{KL}_j$ carried")
axs[1].set_title("(b) unused units collapse to the prior"); axs[1].legend(fontsize=8)

# (c) the latent space
mu=np.load("vae_mu.npy"); y=np.load("vae_y.npy")
U,S,Vt=np.linalg.svd(mu-mu.mean(0),full_matrices=False)
Z=(mu-mu.mean(0))@Vt[:2].T
sc=axs[2].scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=8)
plt.colorbar(sc,ax=axs[2],ticks=range(10)).set_label("digit")
axs[2].set_xlabel("latent 1"); axs[2].set_ylabel("latent 2")
axs[2].set_title(r"(c) the $d_h=10$ latent space, projected")
save(fig,15,"vae_training")

# --- appended: the framework experiments of Section 15.vaelibraries ---------
# Needs cross_check_vae.py, vae_torch.py and vae_tf.py to have been run in
# BookPrograms/chapter15_vae.
RUN = os.environ.get("CH15_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter15_vae"))
L = lambda n: np.load(os.path.join(RUN, n))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) the K-sample bound climbing towards log p
gap = L("gap_table.npy")                    # d_h, L1, L16, L128, L1024, gap
Ks = [1, 16, 128, 1024]
for row, c in zip(gap, ["C0", "C1", "C3"]):
    axs[0].semilogx(Ks, row[1:5], "o-", color=c, ms=5,
                    label=f"$d_h={int(row[0])}$")
axs[0].set_xlabel("$K$, samples in the bound")
axs[0].set_ylabel(r"$\mathcal{L}_K$ (nats)")
axs[0].set_title("(a) the gap of Thm 15.elbo")
axs[0].legend(fontsize=8)
axs[0].set_xticks(Ks)
axs[0].set_xticklabels([str(k) for k in Ks])

# (b) the two gradient estimators
ev = L("estimator_variance.npy")            # d_h, var_rep, var_sf, ratio
axs[1].loglog(ev[:, 0], ev[:, 1], "C0o-", ms=5, label="reparameterisation")
axs[1].loglog(ev[:, 0], ev[:, 2], "C3s--", ms=5, label="score function")
axs[1].set_xlabel("latent dimension $d_h$")
axs[1].set_ylabel("total gradient variance")
axs[1].set_title("(b) why Eq. (15.reparam) is used")
axs[1].legend(fontsize=8)
axs[1].set_xticks(ev[:, 0])
axs[1].set_xticklabels([int(v) for v in ev[:, 0]])

# (c) per-coordinate KL: the collapse cliff
kl = L("kl_per_unit.npy")
axs[2].semilogy(np.arange(1, len(kl) + 1), np.maximum(kl, 1e-6), "C0o-", ms=4)
axs[2].axhline(0.01, color="k", ls=":", lw=1.0)
axs[2].annotate("active threshold", (len(kl) * 0.45, 0.01),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[2].set_xlabel("latent coordinate, ordered")
axs[2].set_ylabel(r"$\mathrm{KL}_j$ (nats)")
axs[2].set_title("(c) posterior collapse, $d_h=50$")

# (d) reconstructions and prior samples
orig, rec, pri = L("orig.npy"), L("recon.npy"), L("prior_samples.npy")
grid = np.zeros((3 * 28, 8 * 28))
for k in range(8):
    grid[:28, k * 28:(k + 1) * 28] = orig[k].reshape(28, 28)
    grid[28:56, k * 28:(k + 1) * 28] = rec[k].reshape(28, 28)
    grid[56:, k * 28:(k + 1) * 28] = pri[k].reshape(28, 28)
axs[3].imshow(grid, cmap="gray_r")
axs[3].set_xticks([])
axs[3].set_yticks([14, 42, 70])
axs[3].set_yticklabels(["data", "reconstruction", "prior sample"], fontsize=8)
axs[3].grid(False)
axs[3].set_title("(d) $d_h=10$")
save(fig, 15, "vae_frameworks")
