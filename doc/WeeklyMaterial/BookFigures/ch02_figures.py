from common import *
rng=np.random.default_rng(2024)

# 1. Gaussian distributions (the chapter's own code)
x=np.linspace(-20,20,400)
fig,ax=plt.subplots(figsize=(5.4,3.4))
for mu,sig in [(0.,1.),(1.,2.),(2.,4.)]:
    ax.plot(x,np.exp(-(x-mu)**2/(2*sig**2))/np.sqrt(2*np.pi*sig**2),label=rf"$\mu={mu:.0f},\ \sigma={sig:.0f}$")
ax.set_xlabel("$x$"); ax.set_ylabel("$p(x)$"); ax.legend(); ax.set_xlim(-15,15)
save(fig,2,"gaussian_distributions")

# 2. central limit theorem
fig,axes=plt.subplots(1,3,figsize=(9.5,2.9),sharey=False)
draws={"uniform":lambda m,r: r.random((20000,m)),
       "exponential":lambda m,r: r.exponential(1.0,(20000,m)),
       "Cauchy":lambda m,r: r.standard_cauchy((20000,m))}
for ax,(nm,f) in zip(axes,draws.items()):
    for m,c in zip([1,10,100],["C0","C1","C2"]):
        z=f(m,np.random.default_rng(1)).mean(1)
        lo,hi=(-4,4) if nm=="Cauchy" else (np.percentile(z,0.2),np.percentile(z,99.8))
        ax.hist(z,bins=90,range=(lo,hi),density=True,histtype="step",lw=1.6,color=c,label=f"$m={m}$")
    ax.set_title(nm,fontsize=10); ax.set_xlabel(r"$\bar{x}$")
    if nm=="Cauchy": ax.set_xlim(-4,4)
axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
save(fig,2,"central_limit_theorem")

# 3. autocorrelation of AR(1)
phi=0.9; N=200000
x=np.empty(N); x[0]=0
e=np.random.default_rng(3).normal(size=N)
for t in range(1,N): x[t]=phi*x[t-1]+e[t]
xc=x-x.mean(); var=xc.var()
d=np.arange(1,60); kap=np.array([np.mean(xc[:N-k]*xc[k:])/var for k in d])
tau=1+2*kap.sum()
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.plot(d,kap,"o-",ms=3,label="measured $\\kappa_d$")
ax.plot(d,phi**d,"k--",label=r"$\phi^d$")
ax.set_xlabel("separation $d$"); ax.set_ylabel(r"$\kappa_d$")
ax.set_title(r"$\phi=0.9$:  $\tau=%.1f$ measured, $(1+\phi)/(1-\phi)=%.0f$"%(tau,(1+phi)/(1-phi)),fontsize=10)
ax.legend()
save(fig,2,"autocorrelation_ar1")

# 4. bias-variance tradeoff (chapter's own experiment)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
np.random.seed(2018)
n,nb,maxd=40,100,14
x=np.linspace(-3,3,n).reshape(-1,1)
y=np.exp(-x**2)+1.5*np.exp(-(x-2)**2)+np.random.normal(0,0.1,x.shape)
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2)
err=np.zeros(maxd); bi=np.zeros(maxd); va=np.zeros(maxd)
for d in range(maxd):
    mdl=make_pipeline(PolynomialFeatures(degree=d),LinearRegression(fit_intercept=False))
    yp=np.empty((yte.shape[0],nb))
    for i in range(nb):
        x_,y_=resample(xtr,ytr); yp[:,i]=mdl.fit(x_,y_).predict(xte).ravel()
    err[d]=np.mean(np.mean((yte-yp)**2,axis=1,keepdims=True))
    bi[d]=np.mean((yte-np.mean(yp,axis=1,keepdims=True))**2)
    va[d]=np.mean(np.var(yp,axis=1,keepdims=True))
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.plot(range(maxd),err,"o-",label="test error")
ax.plot(range(maxd),bi,"s-",label=r"bias$^2$")
ax.plot(range(maxd),va,"^-",label="variance")
ax.set_yscale("log"); ax.set_xlabel("polynomial degree"); ax.set_ylabel("error"); ax.legend()
save(fig,2,"bias_variance_tradeoff")

# 5. Metropolis
def metropolis(log_p,x0,n_steps,step,rng):
    x=np.atleast_1d(np.asarray(x0,float)); chain=np.empty((n_steps,x.size))
    lp=log_p(x); acc=0
    for t in range(n_steps):
        prop=x+step*rng.normal(size=x.size); lpn=log_p(prop)
        if np.log(rng.random())<lpn-lp: x,lp=prop,lpn; acc+=1
        chain[t]=x
    return chain,acc/n_steps
ch,rate=metropolis(lambda v:-0.5*np.sum(v**2),[0.0],100000,2.0,np.random.default_rng(2024))
s=ch[1000:,0]
fig,axes=plt.subplots(1,2,figsize=(9.0,3.2))
axes[0].plot(ch[:600,0],lw=0.8); axes[0].set_xlabel("step $t$"); axes[0].set_ylabel("$x_t$")
axes[0].set_title("first 600 steps of the chain",fontsize=10)
axes[1].hist(s,bins=90,density=True,alpha=0.6,label="Metropolis samples")
g=np.linspace(-4,4,300); axes[1].plot(g,np.exp(-g**2/2)/np.sqrt(2*np.pi),"k-",lw=2,label=r"$\mathcal{N}(0,1)$")
axes[1].set_xlabel("$x$"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=9)
sc=s-s.mean(); kk=[np.mean(sc[:len(sc)-k]*sc[k:])/sc.var() for k in range(1,60)]
axes[1].set_title(r"acceptance %.2f, $\tau=%.1f$"%(rate,1+2*sum(kk)),fontsize=10)
save(fig,2,"metropolis_sampling")
