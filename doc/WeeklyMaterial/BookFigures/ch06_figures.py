from common import *
import warnings, sys; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/kind-inspiring-knuth/mnt/outputs/ch6")
exec(open("/sessions/kind-inspiring-knuth/mnt/outputs/ch6/smo2.py").read().split('if __name__')[0])
from sklearn.datasets import make_blobs, make_moons
np.random.seed(0)

def poly_k(A,B,degree=3,gamma=1.0,coef0=1.0): return (gamma*(A@B.T)+coef0)**degree

# 1. maximum margin and the support vectors
X,y01=make_blobs(n_samples=60,centers=2,cluster_std=0.85,random_state=3); y=np.where(y01==0,-1.,1.)
m=SVC(C=1000.0).fit(X,y); w=(m.lam_sv*m.y_sv)@m.X_sv; b=m.b_
fig,ax=plt.subplots(figsize=(5.4,4.2))
ax.scatter(X[y==-1,0],X[y==-1,1],c="darkblue",s=22,label="class $-1$")
ax.scatter(X[y==1,0],X[y==1,1],c="darkred",s=22,label="class $+1$")
sv=m.sv_
ax.scatter(X[sv,0],X[sv,1],s=180,facecolors="none",edgecolors="k",linewidths=1.6,label="support vectors")
xs=np.linspace(X[:,0].min()-1,X[:,0].max()+1,10)
for off,ls in [(0,"-"),(1,"--"),(-1,"--")]:
    ax.plot(xs,-(w[0]*xs+b-off)/w[1],ls,c="k",lw=1.8 if off==0 else 1.0)
ax.set_xlim(X[:,0].min()-1,X[:,0].max()+1); ax.set_ylim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$"); ax.legend(fontsize=8,loc="best")
ax.set_title(r"margin $2/\|w\|=%.2f$, %d support vectors"%(2/np.linalg.norm(w),sv.sum()),fontsize=10)
save(fig,6,"max_margin_support_vectors")

# 2. effect of C
X,y01=make_blobs(n_samples=120,centers=2,cluster_std=2.6,random_state=7); y=np.where(y01==0,-1.,1.)
fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),sharex=True,sharey=True)
for ax,C in zip(axes,[0.01,0.1,10.0]):
    np.random.seed(0)
    mm=SVC(C=C).fit(X,y); ww=(mm.lam_sv*mm.y_sv)@mm.X_sv; bb=mm.b_
    ax.scatter(X[y==-1,0],X[y==-1,1],c="darkblue",s=14); ax.scatter(X[y==1,0],X[y==1,1],c="darkred",s=14)
    ax.scatter(X[mm.sv_,0],X[mm.sv_,1],s=90,facecolors="none",edgecolors="k",linewidths=1.0)
    xs=np.linspace(X[:,0].min()-1,X[:,0].max()+1,10)
    for off,ls in [(0,"-"),(1,"--"),(-1,"--")]:
        ax.plot(xs,-(ww[0]*xs+bb-off)/ww[1],ls,c="k",lw=1.6 if off==0 else 0.8)
    ax.set_ylim(X[:,1].min()-1,X[:,1].max()+1)
    ax.set_title(r"$C=%g$: %d SV, margin %.2f"%(C,mm.sv_.sum(),2/np.linalg.norm(ww)),fontsize=9.5)
    ax.set_xlabel("$x_0$")
axes[0].set_ylabel("$x_1$")
save(fig,6,"soft_margin_C")

# 3. kernels on the moons
X,y01=make_moons(n_samples=200,noise=0.15,random_state=42); y=np.where(y01==0,-1.,1.)
gx,gy=np.meshgrid(np.linspace(X[:,0].min()-.5,X[:,0].max()+.5,260),
                  np.linspace(X[:,1].min()-.5,X[:,1].max()+.5,260))
G=np.c_[gx.ravel(),gy.ravel()]
fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),sharex=True,sharey=True)
cfgs=[("linear",lambda A,B: A@B.T,1.0),
      ("polynomial, $d=3$",lambda A,B: poly_k(A,B,3,1.0,1.0),5.0),
      ("Gaussian, $\\gamma=1$",lambda A,B: rbf_kernel(A,B,1.0),1.0)]
for ax,(nm,k,C) in zip(axes,cfgs):
    np.random.seed(0)
    mm=SVC(C=C,kernel=k).fit(X,y)
    Z=mm.decision_function(G).reshape(gx.shape)
    ax.contourf(gx,gy,Z,levels=np.linspace(-3,3,25),cmap="RdBu_r",alpha=0.7,extend="both")
    ax.contour(gx,gy,Z,levels=[-1,0,1],colors="k",linewidths=[0.8,1.8,0.8],linestyles=["--","-","--"])
    ax.scatter(X[y==-1,0],X[y==-1,1],c="darkblue",s=12); ax.scatter(X[y==1,0],X[y==1,1],c="darkred",s=12)
    ax.scatter(X[mm.sv_,0],X[mm.sv_,1],s=70,facecolors="none",edgecolors="k",linewidths=0.8)
    ax.set_title("%s\nacc %.3f, %d SV"%(nm,np.mean(mm.predict(X)==y),mm.sv_.sum()),fontsize=9.5)
    ax.set_xlabel("$x_0$"); ax.grid(False)
axes[0].set_ylabel("$x_1$")
save(fig,6,"kernel_decision_boundaries")

# 4. the three losses
t=np.linspace(-3,3,600)
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.plot(t,np.maximum(0,1-t),label=r"hinge, $\max(0,1-yf)$")
ax.plot(t,np.log(1+np.exp(-t))/np.log(2),label=r"logistic, $\log(1+e^{-yf})$")
ax.plot(t,(1-t)**2,label=r"squared, $(1-yf)^2$")
ax.plot(t,(t<0).astype(float),"k:",label="0/1 misclassification")
ax.set_ylim(-0.1,4); ax.set_xlabel(r"margin $yf$"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
ax.axvline(1,ls=":",c="gray",lw=1); ax.text(1.05,3.4,r"$yf=1$",fontsize=8,color="gray")
save(fig,6,"loss_functions")
