from common import *
import warnings, sys; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/kind-inspiring-knuth/mnt/outputs/ch7")
from trees import *
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 1. impurity measures
p=np.linspace(1e-9,1-1e-9,500)
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.plot(p,2*p*(1-p),label="Gini, $2p(1-p)$")
ax.plot(p,-(p*np.log2(p)+(1-p)*np.log2(1-p)),label="entropy")
ax.plot(p,np.minimum(p,1-p),label=r"misclassification, $\min(p,1-p)$")
ax.plot(p,-(p*np.log2(p)+(1-p)*np.log2(1-p))/2,"--",c="0.6",lw=1,label="entropy / 2")
ax.set_xlabel("$p$, proportion of class 1"); ax.set_ylabel("impurity"); ax.legend(fontsize=8)
save(fig,7,"impurity_measures")

# 2. tree decision boundary vs depth
X,y=make_moons(n_samples=400,noise=0.25,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=0)
gx,gy=np.meshgrid(np.linspace(X[:,0].min()-.4,X[:,0].max()+.4,320),
                  np.linspace(X[:,1].min()-.4,X[:,1].max()+.4,320))
G=np.c_[gx.ravel(),gy.ravel()]
fig,axes=plt.subplots(1,4,figsize=(11.5,2.9),sharex=True,sharey=True)
for ax,d in zip(axes,[1,3,5,None]):
    t=DecisionTree("classification","gini",max_depth=d).fit(Xtr,ytr)
    Z=t.predict(G).reshape(gx.shape)
    ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],cmap="RdBu_r",alpha=0.5)
    ax.scatter(Xtr[ytr==0,0],Xtr[ytr==0,1],c="darkblue",s=8)
    ax.scatter(Xtr[ytr==1,0],Xtr[ytr==1,1],c="darkred",s=8)
    ax.set_title("depth %s\ntrain %.3f / test %.3f"%(d if d else r"$\infty$",
        np.mean(t.predict(Xtr)==ytr),np.mean(t.predict(Xte)==yte)),fontsize=9)
    ax.set_xlabel("$x_0$"); ax.grid(False)
axes[0].set_ylabel("$x_1$")
save(fig,7,"tree_depth_boundary")

# 3. variance of an average of correlated predictors
B=np.arange(1,201)
fig,ax=plt.subplots(figsize=(5.4,3.4))
for rho,c in zip([0.0,0.2,0.5,0.8],["C0","C1","C2","C3"]):
    ax.plot(B,rho+(1-rho)/B,c,label=rf"$\rho={rho}$")
    ax.axhline(rho,ls=":",c=c,lw=1)
ax.set_xlabel("number of trees $B$"); ax.set_ylabel(r"$\mathrm{Var}(\bar f)/\sigma^2$")
ax.legend(fontsize=8); ax.set_ylim(0,1.02)
ax.set_title(r"$\rho\sigma^2+(1-\rho)\sigma^2/B$: the floor is $\rho\sigma^2$",fontsize=10)
save(fig,7,"ensemble_variance")

# 4. single tree vs bagging vs forest, and AdaBoost error curve
X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=42)
Bs=[1,3,5,10,25,50,100]
bag=[];frs=[]
for B in Bs:
    bag.append(np.mean(BaggingClassifier(n_estimators=B,rng=np.random.default_rng(1)).fit(Xtr,ytr).predict(Xte)==yte))
    frs.append(np.mean(RandomForestClassifier(n_estimators=B,rng=np.random.default_rng(1)).fit(Xtr,ytr).predict(Xte)==yte))
single=np.mean(DecisionTree("classification").fit(Xtr,ytr).predict(Xte)==yte)
ypm=np.where(ytr==0,-1.,1.); ytem=np.where(yte==0,-1.,1.)
ab=AdaBoost(n_estimators=200,max_depth=1,rng=np.random.default_rng(1)).fit(Xtr,ypm)
Ms=np.arange(1,len(ab.trees_)+1); tr=[];te=[]
ftr=np.zeros(len(ytr)); fte=np.zeros(len(yte))
for a,t in zip(ab.alphas_,ab.trees_):
    ftr=ftr+a*t.predict(Xtr); fte=fte+a*t.predict(Xte)
    tr.append(np.mean(np.sign(ftr)!=ypm)); te.append(np.mean(np.sign(fte)!=ytem))
fig,axes=plt.subplots(1,2,figsize=(9.4,3.4))
axes[0].plot(Bs,bag,"o-",label="bagging"); axes[0].plot(Bs,frs,"s-",label="random forest")
axes[0].axhline(single,ls="--",c="k",lw=1,label="single tree")
axes[0].set_xlabel("number of trees $B$"); axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=8)
axes[1].plot(Ms,tr,label="training error"); axes[1].plot(Ms,te,label="test error")
axes[1].set_xlabel("number of stumps $M$"); axes[1].set_ylabel("error rate"); axes[1].legend(fontsize=8)
axes[1].set_title("AdaBoost with depth-1 trees",fontsize=10)
save(fig,7,"ensembles_and_boosting")
