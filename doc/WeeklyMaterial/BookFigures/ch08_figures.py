from common import *
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/kind-inspiring-knuth/mnt/outputs/ch8")
from nn import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. the XOR problem: why one layer fails
fig,axes=plt.subplots(1,3,figsize=(10.5,3.2))
pts=np.array([[0,0],[0,1],[1,0],[1,1]],float)
for ax,(nm,y) in zip(axes,[("AND",[0,0,0,1]),("OR",[0,1,1,1]),("XOR",[0,1,1,0])]):
    y=np.array(y)
    ax.scatter(pts[y==0,0],pts[y==0,1],s=200,c="darkblue",marker="o",label="0",zorder=3)
    ax.scatter(pts[y==1,0],pts[y==1,1],s=200,c="darkred",marker="s",label="1",zorder=3)
    if nm=="AND": ax.plot([-0.3,1.3],[1.4,0.15],"k--",lw=1.8)
    if nm=="OR":  ax.plot([-0.3,1.3],[0.85,-0.4],"k--",lw=1.8)
    if nm=="XOR":
        ax.plot([-0.3,1.3],[0.85,-0.4],"r:",lw=1.6); ax.plot([-0.3,1.3],[1.4,0.15],"r:",lw=1.6)
        ax.text(0.5,1.28,"no single line works",ha="center",color="crimson",fontsize=9)
    ax.set_xlim(-0.3,1.3); ax.set_ylim(-0.35,1.45); ax.set_title(nm,fontsize=11)
    ax.set_xlabel("$x_1$"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
axes[0].set_ylabel("$x_2$"); axes[0].legend(fontsize=8,loc="lower left")
save(fig,8,"xor_problem")

# 2. activation functions and their derivatives
z=np.linspace(-4,4,600)
fig,axes=plt.subplots(1,2,figsize=(9.4,3.3))
for nm,f,fp in [("sigmoid",sigmoid,sigmoid_prime),("tanh",tanh_,tanh_prime),
                ("ReLU",relu,relu_prime),("ELU",elu,elu_prime)]:
    axes[0].plot(z,f(z),label=nm); axes[1].plot(z,fp(z),label=nm)
axes[0].set_title("activation $f(z)$",fontsize=10); axes[1].set_title("derivative $f'(z)$",fontsize=10)
axes[1].axhline(0.25,ls=":",c="gray",lw=1); axes[1].text(-3.9,0.30,r"$\max\sigma'=1/4$",fontsize=8,color="gray")
for a in axes: a.set_xlabel("$z$"); a.legend(fontsize=8)
save(fig,8,"activation_functions_nn")

# 3. vanishing gradients: gradient norm by layer, sigmoid vs relu
d=load_digits(); Xd=StandardScaler().fit_transform(d.data)
Xtr,Xte,ytr,yte=train_test_split(Xd,d.target,test_size=0.2,random_state=42)
Y=np.zeros((len(ytr),10)); Y[np.arange(len(ytr)),ytr]=1
fig,axes=plt.subplots(1,2,figsize=(9.4,3.3))
depths=[1,2,4,8]
for ax,act in zip(axes,["sigmoid","relu"]):
    for dep in depths:
        sizes=[64]+[30]*dep+[10]
        net=NeuralNetwork(sizes,act,"classification",rng=np.random.default_rng(1))
        gW,_=net._backward(Xtr[:256],Y[:256])
        norms=[np.linalg.norm(g) for g in gW]
        ax.semilogy(range(1,len(norms)+1),norms,"o-",ms=4,label=f"{dep} hidden")
    ax.set_xlabel("layer index $l$"); ax.set_title(act,fontsize=10); ax.legend(fontsize=8)
axes[0].set_ylabel(r"$\|\partial C/\partial W^l\|_F$")
save(fig,8,"vanishing_gradients")

# 4. training curves and the effect of the activation
fig,axes=plt.subplots(1,2,figsize=(9.4,3.3))
for act in ["sigmoid","tanh","relu"]:
    net=NeuralNetwork([64,50,10],act,"classification",gamma=0.1,lmbd=1e-4,epochs=60,
                      batch_size=32,rng=np.random.default_rng(2024)).fit(Xtr,ytr)
    axes[0].semilogy(net.loss_,label=f"{act} (test {np.mean(net.predict(Xte)==yte):.3f})")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training cross entropy"); axes[0].legend(fontsize=8)
axes[0].set_title("64-50-10 on the digits",fontsize=10)
units=[2,5,10,20,50,100,200]; tr=[];te=[]
for u in units:
    net=NeuralNetwork([64,u,10],"relu","classification",gamma=0.1,lmbd=1e-4,epochs=60,
                      batch_size=32,rng=np.random.default_rng(2024)).fit(Xtr,ytr)
    tr.append(np.mean(net.predict(Xtr)==ytr)); te.append(np.mean(net.predict(Xte)==yte))
axes[1].semilogx(units,tr,"o-",label="training"); axes[1].semilogx(units,te,"s-",label="test")
axes[1].set_xlabel("hidden units"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=8)
axes[1].set_title("capacity",fontsize=10)
save(fig,8,"digits_training")
from common import *
from scipy.special import erf
sig=lambda z:1/(1+np.exp(-z))
Phi=lambda z:0.5*(1+erf(z/np.sqrt(2)))
gelu=lambda z:z*Phi(z)
gelu_t=lambda z:0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
swish=lambda z:z*sig(z)
mish=lambda z:z*np.tanh(np.log1p(np.exp(np.clip(z,-30,30))))
softplus=lambda z:np.log1p(np.exp(np.clip(z,-30,30)))
relu=lambda z:np.maximum(0,z)
lrelu=lambda z:np.where(z>0,z,0.01*z)
elu=lambda z:np.where(z>0,z,np.exp(np.minimum(z,0))-1)

z=np.linspace(-4,4,1200); h=1e-5
d=lambda f: (f(z+h)-f(z-h))/(2*h)

fig,axes=plt.subplots(2,3,figsize=(11.0,6.0))
fams=[("saturating",[("sigmoid",sig),("tanh",np.tanh)]),
      ("rectified",[("ReLU",relu),("leaky ReLU",lrelu),("ELU",elu)]),
      ("smooth gated",[("GELU",gelu),("Swish/SiLU",swish),("Mish",mish),("softplus",softplus)])]
for col,(title,fs) in enumerate(fams):
    for nm,f in fs:
        axes[0,col].plot(z,f(z),label=nm); axes[1,col].plot(z,d(f),label=nm)
    axes[0,col].set_title(title,fontsize=11); axes[0,col].legend(fontsize=8)
    axes[1,col].legend(fontsize=8); axes[1,col].set_xlabel("$z$")
    axes[0,col].set_ylim(-1.5,4); axes[1,col].set_ylim(-0.15,1.35)
axes[1,0].axhline(0.25,ls=":",c="gray",lw=1)
axes[1,0].text(-3.9,0.30,r"$\max\sigma'=1/4$",fontsize=8,color="gray")
for c in (1,2): axes[1,c].axhline(1.0,ls=":",c="gray",lw=1)
axes[1,2].text(-3.9,1.14,r"$f'>1$ possible",fontsize=8,color="gray")
axes[0,0].set_ylabel("$f(z)$"); axes[1,0].set_ylabel("$f'(z)$")
save(fig,8,"activation_families")

# GELU in detail: exact vs approximation, and the non-monotone dip
fig,axes=plt.subplots(1,2,figsize=(9.4,3.3))
zz=np.linspace(-4,4,1200)
axes[0].plot(zz,relu(zz),"k--",lw=1.2,label="ReLU")
for nm,f in [("GELU",gelu),("Swish",swish),("Mish",mish)]:
    axes[0].plot(zz,f(zz),label=nm)
axes[0].set_xlim(-3,2); axes[0].set_ylim(-0.45,2.05)
axes[0].set_xlabel("$z$"); axes[0].set_ylabel("$f(z)$"); axes[0].legend(fontsize=8)
for nm,f,c in [("GELU",gelu,"C0"),("Swish",swish,"C1"),("Mish",mish,"C2")]:
    i=np.argmin(f(zz)); axes[0].plot(zz[i],f(zz)[i],"o",c=c,ms=5)
axes[0].set_title("the negative dip: minima marked",fontsize=10)
zf=np.linspace(-8,8,4000)
axes[1].semilogy(zf,np.abs(gelu(zf)-gelu_t(zf)),lw=1.5)
axes[1].set_xlabel("$z$"); axes[1].set_ylabel(r"$|\mathrm{GELU}-\mathrm{approx}|$")
axes[1].set_title(r"error of the $\tanh$ approximation (8.geluapprox)",fontsize=10)
axes[1].axhline(np.abs(gelu(zf)-gelu_t(zf)).max(),ls=":",c="crimson",lw=1)
axes[1].text(-7.7,np.abs(gelu(zf)-gelu_t(zf)).max()*1.4,
             r"max $%.1f\times10^{-4}$"%(np.abs(gelu(zf)-gelu_t(zf)).max()*1e4),fontsize=8,color="crimson")
save(fig,8,"gelu_detail")

# activation comparison at depth: the measured table as a figure
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/kind-inspiring-knuth/mnt/outputs/ch8")
from nn import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
d=load_digits(); Xs=StandardScaler().fit_transform(d.data)
Xtr,Xte,ytr,yte=train_test_split(Xs,d.target,test_size=0.2,random_state=42)
acts=["sigmoid","tanh","relu","leaky_relu","elu","gelu","swish","mish"]
mean=[];sd=[]
for act in acts:
    a=[np.mean(NeuralNetwork([64,30,30,30,10],act,"classification",gamma=0.1,lmbd=1e-4,
        epochs=60,batch_size=32,rng=np.random.default_rng(s)).fit(Xtr,ytr).predict(Xte)==yte)
        for s in range(5)]
    mean.append(np.mean(a)); sd.append(np.std(a))
fig,ax=plt.subplots(figsize=(6.4,3.4))
xs=np.arange(len(acts))
ax.errorbar(xs,mean,yerr=sd,fmt="o",capsize=4,ms=6)
ax.axhspan(min(m for m,a in zip(mean,acts) if a!="sigmoid")-0.002,
           max(mean)+0.002,color="C0",alpha=0.10)
ax.set_xticks(xs); ax.set_xticklabels(acts,rotation=30,ha="right")
ax.set_ylabel("test accuracy"); ax.grid(axis="y",alpha=0.3)
ax.set_title("three hidden layers, 5 seeds: only the sigmoid separates",fontsize=10)
save(fig,8,"activation_comparison")

