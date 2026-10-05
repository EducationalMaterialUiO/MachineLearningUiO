import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter14_boltzmann"))
from common import plt, np, save
import rbm

fig,axs=plt.subplots(1,3,figsize=(13,3.8))

# (a) CD-k bias
M,N=8,4
P=rbm.init_rbm(M,N,np.random.default_rng(1),scale=1.0)
base=(np.random.default_rng(2).random((4,M))<0.5).astype(float)
X=base[np.random.default_rng(3).integers(0,4,200)]
flat=lambda d: np.concatenate([np.ravel(d[k]) for k in ("W","a","b")])
ge=flat(rbm.exact_gradient(P,X))
ks=[1,2,3,5,8,12,20,35,60,100]; rel=[]; cos=[]
for k in ks:
    acc=None
    for r in range(40):
        gk=flat(rbm.cd_gradient(P,X,k=k,rng=np.random.default_rng(1000+r)))
        acc=gk if acc is None else acc+gk
    gk=acc/40
    rel.append(np.linalg.norm(gk-ge)/np.linalg.norm(ge))
    cos.append(gk@ge/(np.linalg.norm(gk)*np.linalg.norm(ge)))
axs[0].loglog(ks,rel,"C0o-",label=r"$\|\hat g_k-g\|/\|g\|$")
axs[0].loglog(ks,1-np.array(cos),"C1s-",label=r"$1-\cos(\hat g_k,g)$")
axs[0].set_xlabel("$k$, Gibbs sweeps per update"); axs[0].set_ylabel("error")
axs[0].set_title("(a) contrastive divergence is biased"); axs[0].legend(fontsize=8)

# (b) Gibbs convergence
M2=6
P2=rbm.init_rbm(M2,3,np.random.default_rng(1),scale=1.2)
V=rbm.all_states(M2); logp=-rbm.free_energy(P2,V); logp-=np.logaddexp.reduce(logp)
pex=np.exp(logp); idx={tuple(v):i for i,v in enumerate(V.astype(int))}
rng=np.random.default_rng(0); Xs=(rng.random((500,M2))<0.5).astype(float)
counts=np.zeros(len(V)); tvs=[]; ns=[]
for t in range(4000):
    Xs,_,_,_=rbm.gibbs_step(P2,Xs,rng)
    if t>=100:
        for row in Xs.astype(int): counts[idx[tuple(row)]]+=1
        if (t-100)%100==0 and t>100:
            tvs.append(0.5*np.abs(counts/counts.sum()-pex).sum()); ns.append(counts.sum())
axs[1].loglog(ns,tvs,"C0o-",ms=3,label="block Gibbs")
axs[1].loglog(ns,tvs[0]*np.sqrt(ns[0]/np.array(ns)),"k--",lw=1,label=r"$\propto n^{-1/2}$")
axs[1].axhline(0.3601,color="C3",ls=":",label="a uniform guess")
axs[1].set_xlabel("samples drawn"); axs[1].set_ylabel(r"$\|\hat p-p\|_{\mathrm{TV}}$")
axs[1].set_title("(b) Gibbs samples the model"); axs[1].legend(fontsize=8)

# (c) training curves
H=np.load("bas_hist.npy"); it=np.load("bas_it.npy"); D=np.load("bas_D.npy")
for row,lab,c in zip(H,["CD-1","CD-10","exact gradient"],["C0","C1","C2"]):
    axs[2].plot(it,row,c,lw=1.5,label=lab)
axs[2].axhline(np.log(1/len(D)),color="k",ls="--",lw=1,label="perfect model")
axs[2].set_xlabel("update"); axs[2].set_ylabel("log-likelihood per image")
axs[2].set_title("(c) what the bias costs"); axs[2].legend(fontsize=8,loc="lower right")
save(fig,14,"rbm_training")

# --- appended: the framework experiments of Section 14.rbmlibraries ---------
# Needs cross_check_rbm.py, rbm_torch.py and rbm_tf.py to have been run in
# BookPrograms/chapter14_boltzmann.
RUN = os.environ.get("CH14_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter14_boltzmann"))
L = lambda n: np.load(os.path.join(RUN, n))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) pseudo-likelihood against epoch
for nm, f, c in [("CD-1", "hist_cd1.npy", "C0"),
                 ("CD-10", "hist_cd10.npy", "C1"),
                 ("PCD-1", "hist_pcd1.npy", "C2")]:
    h = L(f)
    axs[0].plot(h[:, 0], h[:, 1], "o-", color=c, ms=5, label=nm)
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("pseudo-likelihood per image")
axs[0].set_title("(a) MNIST, three training signals")
axs[0].legend(fontsize=9)

# (b) the exact gradient against autograd
errs = L("exact_grad_errors.npy")
names = ["$W$", "$a$", "$b$"]
axs[1].bar(np.arange(len(errs)), errs, 0.5, color="C0")
axs[1].axhline(np.finfo(float).eps, color="k", ls=":", lw=1.0)
axs[1].annotate(r"$\epsilon_{\rm mach}$", (len(errs) - 1.6, np.finfo(float).eps),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[1].set_yscale("log")
axs[1].set_xticks(np.arange(len(errs)))
axs[1].set_xticklabels(names)
axs[1].set_ylabel("max gradient difference")
axs[1].set_title("(b) Thm 14.gradient vs autograd")
axs[1].set_ylim(1e-18, 1e-12)

# (c) learned filters
filt = L("filters.npy")                       # (64, 784)
grid = np.zeros((4 * 28, 4 * 28))
for k in range(16):
    r, c = divmod(k, 4)
    grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = filt[k].reshape(28, 28)
v = np.abs(grid).max()
axs[2].imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v)
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].grid(False)
axs[2].set_title("(c) sixteen learned filters")

# (d) a Gibbs chain from noise
S = L("samples.npy")                          # (4, 8, 784)
lab = ["1", "10", "100", "1000"]
grid = np.zeros((4 * 28, 6 * 28))
for r in range(4):
    for c in range(6):
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = S[r, c].reshape(28, 28)
axs[3].imshow(grid, cmap="gray_r")
axs[3].set_xticks([])
axs[3].set_yticks([(r + 0.5) * 28 for r in range(4)])
axs[3].set_yticklabels([f"{t} sweeps" for t in lab], fontsize=8)
axs[3].grid(False)
axs[3].set_title("(d) Gibbs chain from noise")
save(fig, 14, "rbm_frameworks")
