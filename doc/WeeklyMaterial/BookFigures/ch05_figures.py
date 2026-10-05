from common import *
import warnings; warnings.filterwarnings("ignore")

# 1. activation functions
z=np.linspace(-6,6,600)
sig=1/(1+np.exp(-z))
fig,axes=plt.subplots(1,3,figsize=(10.0,2.8))
axes[0].plot(z,sig); axes[0].set_title("sigmoid $\\sigma(t)$",fontsize=10)
axes[1].plot(z,np.where(z>=0,1.0,0.0)); axes[1].set_title("step",fontsize=10)
axes[2].plot(z,np.tanh(z)); axes[2].set_title(r"$\tanh$",fontsize=10)
for a in axes: a.set_xlabel("$t$")
axes[0].plot(z,sig*(1-sig),"--",label=r"$\sigma'=\sigma(1-\sigma)$"); axes[0].legend(fontsize=8)
axes[0].annotate(r"max $=1/4$",xy=(0,0.25),xytext=(1.6,0.42),fontsize=8,
                 arrowprops=dict(arrowstyle="->",lw=0.9))
save(fig,5,"activation_functions")

# 2. why not squared error: gradient magnitudes
t=np.linspace(-6,6,600); p=1/(1+np.exp(-t)); y=1.0
g_ce=-(y-p)                      # d/dt of cross entropy
g_sq=-2*(y-p)*p*(1-p)            # d/dt of squared error
fig,axes=plt.subplots(1,2,figsize=(9.0,3.2))
axes[0].plot(t,-(y*np.log(p)+(1-y)*np.log(1-p)),label="cross entropy")
axes[0].plot(t,(y-p)**2,label="squared error")
axes[0].set_xlabel("$t$"); axes[0].set_ylabel("loss, for $y=1$"); axes[0].legend(fontsize=9)
axes[1].plot(t,np.abs(g_ce),label=r"$|\partial C/\partial t|$, cross entropy")
axes[1].plot(t,np.abs(g_sq),label=r"$|\partial C/\partial t|$, squared error")
axes[1].axvspan(-6,-2,color="crimson",alpha=0.10)
axes[1].text(-5.6,0.62,"confidently\nwrong",fontsize=8,color="crimson")
axes[1].set_xlabel("$t$"); axes[1].set_ylabel("gradient magnitude"); axes[1].legend(fontsize=8)
save(fig,5,"crossentropy_vs_squared")

# 3. decision boundary + probability contours
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
X,yb=make_blobs(n_samples=200,centers=2,cluster_std=1.8,random_state=5)
clf=LogisticRegression().fit(X,yb)
gx,gy=np.meshgrid(np.linspace(X[:,0].min()-1,X[:,0].max()+1,300),
                  np.linspace(X[:,1].min()-1,X[:,1].max()+1,300))
P=clf.predict_proba(np.c_[gx.ravel(),gy.ravel()])[:,1].reshape(gx.shape)
fig,ax=plt.subplots(figsize=(5.2,4.0))
c=ax.contourf(gx,gy,P,levels=np.linspace(0,1,21),cmap="RdBu_r",alpha=0.75)
ax.contour(gx,gy,P,levels=[0.5],colors="k",linewidths=2)
ax.contour(gx,gy,P,levels=[0.1,0.9],colors="k",linewidths=0.8,linestyles="--")
ax.scatter(X[yb==0,0],X[yb==0,1],c="darkblue",s=16,edgecolors="w",linewidths=0.4)
ax.scatter(X[yb==1,0],X[yb==1,1],c="darkred",s=16,edgecolors="w",linewidths=0.4)
fig.colorbar(c,ax=ax,label=r"$p(y=1\mid x)$"); ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$"); ax.grid(False)
save(fig,5,"logistic_decision_boundary")

# 4. Wisconsin: ROC and confusion
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix
c=load_breast_cancer()
Xtr,Xte,ytr,yte=train_test_split(c.data,c.target,random_state=0)
m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)).fit(Xtr,ytr)
pr=m.predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,pr)
fig,axes=plt.subplots(1,2,figsize=(9.0,3.6))
axes[0].plot(fpr,tpr,lw=2,label=f"AUC = {roc_auc_score(yte,pr):.4f}")
axes[0].plot([0,1],[0,1],"k--",lw=1,label="chance")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].legend(fontsize=9)
axes[0].set_title("ROC, Wisconsin test set",fontsize=10)
cm=confusion_matrix(yte,m.predict(Xte))
axes[1].imshow(cm,cmap="Blues"); axes[1].grid(False)
for i in range(2):
    for j in range(2):
        axes[1].text(j,i,cm[i,j],ha="center",va="center",fontsize=15,
                     color="w" if cm[i,j]>cm.max()/2 else "k")
axes[1].set_xticks([0,1],["malignant","benign"]); axes[1].set_yticks([0,1],["malignant","benign"])
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true"); axes[1].set_title("confusion matrix",fontsize=10)
save(fig,5,"wisconsin_roc_confusion")
