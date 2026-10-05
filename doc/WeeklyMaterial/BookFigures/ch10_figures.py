import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter10_convolutional_networks"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import plt, np, save
import cnn
from sklearn.datasets import load_digits

# ===================== Figure 1: convolution as a sparse matrix =============
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
rng=np.random.default_rng(1)

# (a) 1-D Toeplitz
a=np.array([2.,-1.,3.]); n=8
T=np.zeros((n+len(a)-1,n))
for j in range(n): T[j:j+len(a),j]=a
im=axs[0].imshow(T,cmap="RdBu_r",vmin=-3,vmax=3)
axs[0].set_title(r"(a) 1-D convolution: Toeplitz $\mathbf{T}_w$")
axs[0].set_xlabel("input index"); axs[0].set_ylabel("output index"); axs[0].grid(False)
plt.colorbar(im,ax=axs[0],fraction=0.046)

# (b) doubly block Toeplitz for 5x5 image, 3x3 kernel, S=1, P=0
W=rng.normal(size=(1,1,3,3)); H=5
Wp=np.zeros((9,25))
for k in range(25):
    e=np.zeros(25); e[k]=1
    Yk,_=cnn.conv_forward(e.reshape(1,1,H,H),W,np.zeros(1),1,0)
    Wp[:,k]=Yk.ravel()
im=axs[1].imshow(Wp,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
axs[1].set_title(r"(b) 2-D: doubly block Toeplitz $\mathbf{W}'$")
axs[1].set_xlabel(r"$\mathrm{vec}(\mathbf{X})$ index"); axs[1].set_ylabel(r"$\mathrm{vec}(\mathbf{Y})$ index")
axs[1].grid(False); plt.colorbar(im,ax=axs[1],fraction=0.046)
nz=(Wp!=0).sum()
axs[1].text(0.98,0.03,f"{nz}/{Wp.size} nonzero\n{len(set(np.round(Wp[Wp!=0],9)))} distinct values",
            transform=axs[1].transAxes,ha="right",va="bottom",fontsize=8,
            bbox=dict(fc="white",alpha=0.85,ec="none"))

# (c) parameter counts
sizes=np.array([8,16,32,64,128,256,512])
dense=(sizes**2*3)*(sizes**2*3)      # one fully connected layer, same size out
conv=3*3*3*64+64                     # 64 filters of 3x3x3
axs[2].loglog(sizes,dense,"o-",label="dense layer, same size out")
axs[2].axhline(conv,color="C1",ls="--",label=r"conv $3\times3\times3$, $K=64$")
axs[2].set_xlabel("image side (pixels, 3 channels)"); axs[2].set_ylabel("parameters")
axs[2].set_title("(c) why dense layers do not scale"); axs[2].legend(fontsize=8)
save(fig,10,"convolution_structure")

# ===================== Figure 2: equivariance ==============================
d=load_digits(); img=d.images[0]/16.0
canvas=np.zeros((1,1,14,14)); canvas[0,0,2:10,2:10]=img
K=np.array([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]]])   # Sobel-x
fig,axs=plt.subplots(2,4,figsize=(12,6))
for col,s in enumerate([0,2,4,6]):
    Xs=np.roll(canvas,s,axis=3)
    Y,_=cnn.conv_forward(Xs,K,np.zeros(1),1,1)
    axs[0,col].imshow(Xs[0,0],cmap="gray"); axs[0,col].set_title(f"input, shift $s={s}$",fontsize=10)
    axs[1,col].imshow(Y[0,0],cmap="RdBu_r"); axs[1,col].set_title(f"feature map",fontsize=10)
    for ax in (axs[0,col],axs[1,col]): ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(r"Translation equivariance: $\;\mathcal{C}_w(T_s x)=T_s\,\mathcal{C}_w(x)$"
             "\n(the feature map slides with the input, unchanged in form)",fontsize=11)
save(fig,10,"equivariance")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter10_convolutional_networks"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import plt, np, save
import cnn

fig,axs=plt.subplots(1,3,figsize=(13,3.8))

# (a) learning curves
hc=np.load("hist_cnn.npy"); hd=np.load("hist_dense.npy")
axs[0].plot(hc[:,0],hc[:,2],"C0o-",ms=3,label="CNN, 1898 par.")
axs[0].plot(hd[:,0],hd[:,2],"C1s-",ms=3,label="dense, 1885 par.")
axs[0].set_xlabel("epoch"); axs[0].set_ylabel("test accuracy")
axs[0].set_title(r"(a) centred $8\times8$ digits: a tie"); axs[0].legend(fontsize=8,loc="lower right")

# (b) the shift experiment
labels=["centred\n$8\\times8$","random offset\n$12\\times12$"]
cnn_m=[0.9656,0.8774]; cnn_lo=[0.9574,0.8574]; cnn_hi=[0.9704,0.9185]
den_m=[0.9670,0.7070]; den_lo=[0.9593,0.6815]; den_hi=[0.9704,0.7315]
x=np.arange(2); w=0.35
axs[1].bar(x-w/2,cnn_m,w,yerr=[np.array(cnn_m)-cnn_lo,np.array(cnn_hi)-cnn_m],
           capsize=4,label="CNN",color="C0")
axs[1].bar(x+w/2,den_m,w,yerr=[np.array(den_m)-den_lo,np.array(den_hi)-den_m],
           capsize=4,label="dense",color="C1")
for xi,(c,d) in enumerate(zip(cnn_m,den_m)):
    axs[1].text(xi-w/2,c+0.055,f"{c:.3f}",ha="center",fontsize=8)
    axs[1].text(xi+w/2,d+0.055,f"{d:.3f}",ha="center",fontsize=8)
axs[1].set_xticks(x); axs[1].set_xticklabels(labels,fontsize=9)
axs[1].set_ylim(0,1.12); axs[1].set_ylabel("test accuracy, 5 seeds")
axs[1].set_title("(b) translation is where it shows"); axs[1].legend(fontsize=8,loc="lower left")

# (c) learned first-layer filters
P=np.load("params_cnn.npz"); W1=P["W1"]
K=W1.shape[0]; vmax=np.abs(W1).max()
big=np.ones((2*4-1,(K//2)*4-1))*np.nan
for k in range(K):
    r,c=divmod(k,K//2)
    big[r*4:r*4+3,c*4:c*4+3]=W1[k,0]
im=axs[2].imshow(big,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
axs[2].set_title(r"(c) the eight learned $3\times3$ filters")
axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].grid(False)
plt.colorbar(im,ax=axs[2],fraction=0.046)
save(fig,10,"cnn_results")
# --- appended to ch10_figures.py -------------------------------------------
# ===================== Figure 4: the frameworks =============================
# Needs cross_check.py, cnn_torch.py and cnn_tf.py to have been run in
# BookPrograms/chapter10_convolutional_networks.  Set CH10_RUN to that
# directory if the figures are built from elsewhere.
RUN = os.environ.get("CH10_RUN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter10_convolutional_networks"))
L = lambda n: np.load(os.path.join(RUN, n))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))

# (a) test accuracy against epoch, both frameworks, both heads
ht, hk = L("hist_torch.npy"), L("hist_keras.npy")
htg, hkg = L("hist_torch_gap.npy"), L("hist_keras_gap.npy")
axs[0].plot(ht[:, 0], ht[:, 2], "C0o-", ms=4, label="PyTorch, dense head")
axs[0].plot(hk[:, 0], hk[:, 2], "C1s--", ms=4, label="TensorFlow, dense head")
axs[0].plot(htg[:, 0], htg[:, 2], "C2o-", ms=4, label="PyTorch, GAP head")
axs[0].plot(hkg[:, 0], hkg[:, 2], "C3s--", ms=4, label="TensorFlow, GAP head")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("test accuracy")
axs[0].set_title("(a) MNIST, the two frameworks")
axs[0].legend(fontsize=7.5, loc="lower right")

# (b) the three implementations agree to rounding
rows = L("crosscheck_rows.npy")               # (6, 3): torch err, keras err, scale
names = ["$W_1$", "$b_1$", "$W_2$", "$b_2$", "$W_3$", "$b_3$"]
idx = np.arange(len(names))
axs[1].bar(idx - 0.2, rows[:, 0], 0.38, label="ours vs PyTorch", color="C0")
axs[1].bar(idx + 0.2, rows[:, 1], 0.38, label="ours vs TensorFlow", color="C1")
axs[1].axhline(np.finfo(float).eps, color="k", ls=":", lw=1.0)
axs[1].annotate(r"$\epsilon_{\rm mach}$", (len(names) - 1.4, np.finfo(float).eps),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
axs[1].set_yscale("log")
axs[1].set_xticks(idx)
axs[1].set_xticklabels(names)
axs[1].set_ylabel("max gradient difference")
axs[1].set_title("(b) the same gradient, three ways")
axs[1].legend(fontsize=8)
axs[1].set_ylim(1e-18, 1e-12)

# (c) where the parameters are
pc = L("paramcounts.npy")
lab = ["conv\n$1\\!\\to\\!32$", "conv\n$32\\!\\to\\!64$",
       "dense\n$3136\\!\\to\\!1024$", "dense\n$1024\\!\\to\\!10$"]
vals = [pc[0], pc[2], pc[5], pc[7]]
bars = axs[2].bar(range(4), vals, color=["C0", "C0", "C3", "C3"])
axs[2].set_yscale("log")
axs[2].set_xticks(range(4))
axs[2].set_xticklabels(lab, fontsize=8)
axs[2].set_ylabel("parameters")
axs[2].set_title("(c) Eq. (10.paramcount) layer by layer")
for b, v in zip(bars, vals):
    axs[2].annotate(f"{int(v)}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", textcoords="offset points", xytext=(0, 3),
                    fontsize=8)
axs[2].set_ylim(1e2, 2e7)

# (d) the first-layer filters of the trained PyTorch model
filt = L("filters_torch.npy")                 # (32, 3, 3)
grid = np.ones((4 * 4 - 1, 8 * 4 - 1)) * np.nan
for k in range(32):
    r, c = divmod(k, 8)
    grid[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = filt[k]
v = np.nanmax(np.abs(grid))
axs[3].imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v)
axs[3].set_xticks([])
axs[3].set_yticks([])
axs[3].grid(False)
axs[3].set_title("(d) the 32 trained $3\\times3$ filters")
save(fig, 10, "cnn_frameworks")
