from common import *
rng=np.random.default_rng(2024)

# 1. condition number of the Vandermonde matrix
degs=np.arange(1,15); k1=[];k2=[]
x=np.linspace(0,1,100)
for d in degs:
    X=np.vander(x,d+1,increasing=True)
    k1.append(np.linalg.cond(X)); k2.append(np.linalg.cond(X.T@X))
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.semilogy(degs,k1,"o-",label=r"$\kappa_2(X)$")
ax.semilogy(degs,k2,"s-",label=r"$\kappa_2(X^TX)=\kappa_2(X)^2$")
ax.axhline(1/np.finfo(float).eps,ls="--",c="k",lw=1)
ax.text(1.2,1/np.finfo(float).eps*1.6,r"$1/\varepsilon_{\rm mach}$",fontsize=9)
ax.set_xlabel("polynomial degree"); ax.set_ylabel("condition number"); ax.legend()
save(fig,1,"vandermonde_conditioning")

# 2. singular values: direct SVD vs via X^T X
n,p=60,12
U,_=np.linalg.qr(rng.normal(size=(n,p))); V,_=np.linalg.qr(rng.normal(size=(p,p)))
s_true=np.logspace(0,-7,p); X=(U*s_true)@V.T
s_svd=np.linalg.svd(X,compute_uv=False)
s_gram=np.sqrt(np.maximum(np.linalg.eigvalsh(X.T@X)[::-1],0))
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.semilogy(range(1,p+1),s_true,"k-",lw=2,label="exact")
ax.semilogy(range(1,p+1),s_svd,"o",label="from the SVD")
ax.semilogy(range(1,p+1),s_gram,"s",mfc="none",label=r"from eigenvalues of $X^TX$")
ax.set_xlabel("index $i$"); ax.set_ylabel(r"singular value $\sigma_i$"); ax.legend()
save(fig,1,"singular_values_accuracy")

# 3. low-rank approximation
N=256; xx,yy=np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
A=np.exp(-(xx**2+yy**2)/2)+0.6*np.sin(3*xx)*np.cos(2*yy)+0.3*np.exp(-((xx-1.5)**2+(yy+1)**2))
U,s,Vt=np.linalg.svd(A)
fig,axes=plt.subplots(1,4,figsize=(9.5,2.6))
for ax,chi in zip(axes,[1,5,20,N]):
    Ak=(U[:,:chi]*s[:chi])@Vt[:chi]
    ax.imshow(Ak,cmap="viridis"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    err=np.linalg.norm(A-Ak)/np.linalg.norm(A)
    ax.set_title((r"$\chi=%d$"%chi)+("\n(exact)" if chi==N else "\nrel. err %.1e"%err),fontsize=9)
save(fig,1,"low_rank_approximation")

fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.semilogy(np.arange(1,61),s[:60]/s[0],"o-",ms=3)
frac=np.cumsum(s**2)/np.sum(s**2)
ax2=ax.twinx(); ax2.plot(np.arange(1,61),frac[:60],"r--"); ax2.set_ylabel("cumulative fraction of $\\|A\\|_F^2$",color="r"); ax2.grid(False)
ax.set_xlabel(r"index $k$"); ax.set_ylabel(r"$\sigma_k/\sigma_0$")
save(fig,1,"singular_value_spectrum")

# 4. PCA
mean=[0,0]; C=np.array([[3.0,2.0],[2.0,2.0]])
D=rng.multivariate_normal(mean,C,600)
Dc=D-D.mean(0); U2,s2,V2t=np.linalg.svd(Dc,full_matrices=False)
lam=s2**2/(len(D)-1)
fig,ax=plt.subplots(figsize=(4.6,4.2))
ax.scatter(D[:,0],D[:,1],s=8,alpha=0.35)
for i in range(2):
    v=V2t[i]*np.sqrt(lam[i])*2.5
    ax.annotate("",xy=Dc.mean(0)+v+D.mean(0),xytext=D.mean(0),
                arrowprops=dict(arrowstyle="->",lw=2.2,color=["crimson","darkorange"][i]))
    ax.text(*(D.mean(0)+v*1.12),r"$v_%d$"%i,color=["crimson","darkorange"][i],fontsize=11)
ax.set_aspect("equal"); ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
ax.set_title(r"explained variance: %.2f, %.2f"%tuple(lam/lam.sum()),fontsize=10)
save(fig,1,"pca_principal_axes")
