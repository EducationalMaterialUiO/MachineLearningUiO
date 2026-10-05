
# 5. following the steps with and without momentum: a quartic with two minima, and a bowl with kappa=10
def quartic(x): return x**4-3.0*x**2+x
def quartic_grad(x): return 4.0*x**3-6.0*x+1.0
def descend(grad,x0,gamma,beta=0.0,steps=40):
    x=np.array(x0,dtype=float); v=np.zeros_like(x); P=[x.copy()]
    for _ in range(steps):
        v=beta*v+gamma*grad(x); x=x-v; P.append(x.copy())
    return np.array(P)
fig,axes=plt.subplots(1,2,figsize=(10.0,3.6))
ax=axes[0]; xs=np.linspace(-2.2,2.2,400); ax.plot(xs,quartic(xs),color="0.5",lw=1.4)
for beta,c,lab in [(0.0,"C0",r"plain, $\gamma=0.05$"),(0.7,"crimson",r"momentum, $\gamma=0.05$, $\beta=0.7$")]:
    P=descend(quartic_grad,2.0,0.05,beta); ax.plot(P,quartic(P),"o-",ms=3,lw=1,color=c,alpha=0.85,label=lab)
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)=x^4-3x^2+x$"); ax.set_ylim(-4.5,8); ax.legend(fontsize=8,loc="upper center")
ax.set_title("start at $x_0=2$: same learning rate, one extra line",fontsize=9.5)
ax=axes[1]; lam2=np.array([1.0,10.0]); bowl_grad=lambda t: lam2*t
t1,t2=np.meshgrid(np.linspace(-1.2,2.5,200),np.linspace(-1.6,1.6,200))
ax.contour(t1,t2,0.5*(t1**2+10*t2**2),levels=np.geomspace(0.02,20,14),colors="0.75",linewidths=0.7)
for gamma,beta,c,lab in [(0.04,0.0,"C1",r"plain, $\gamma=0.04$"),(0.18,0.0,"C0",r"plain, $\gamma=0.18=0.9\,\gamma_{\max}$"),(0.18,0.3,"crimson",r"momentum, $\gamma=0.18$, $\beta=0.3$")]:
    P=descend(bowl_grad,np.array([2.0,1.4]),gamma,beta); ax.plot(P[:,0],P[:,1],"o-",ms=2.5,lw=1,color=c,label=lab)
ax.plot(0,0,"k*",ms=9); ax.set_xlabel(r"$\theta_1$ ($\lambda_1=1$)"); ax.set_ylabel(r"$\theta_2$ ($\lambda_2=10$)")
ax.set_title(r"$C=\frac{1}{2}(\theta_1^2+10\,\theta_2^2)$, $\kappa=10$, 40 steps each",fontsize=9.5); ax.legend(fontsize=8,loc="lower right"); ax.set_aspect("equal")
save(fig,4,"gd_steps")

# 6. adaptive methods: axis-aligned bowl against the same bowl rotated by 45 degrees
def optimiser_step(method,theta,g,state,t,gamma,beta=0.9,rho=0.99,beta1=0.9,beta2=0.999,eps=1e-8):
    if method=="plain": return theta-gamma*g,state
    if method=="momentum":
        state["v"]=v=beta*state.get("v",0.0)+gamma*g; return theta-v,state
    if method=="adagrad":
        state["r"]=r=state.get("r",0.0)+g*g; return theta-gamma*g/(np.sqrt(r)+eps),state
    if method=="rmsprop":
        state["r"]=r=rho*state.get("r",0.0)+(1-rho)*g*g; return theta-gamma*g/(np.sqrt(r)+eps),state
    if method=="adam":
        state["m"]=m=beta1*state.get("m",0.0)+(1-beta1)*g; state["r"]=r=beta2*state.get("r",0.0)+(1-beta2)*g*g
        return theta-gamma*(m/(1-beta1**t))/(np.sqrt(r/(1-beta2**t))+eps),state
def optimise(grad,theta0,method,gamma,steps):
    th=np.array(theta0,dtype=float); st={}; P=[th.copy()]
    for t in range(1,steps+1):
        th,st=optimiser_step(method,th,grad(th),st,t,gamma); P.append(th.copy())
    return np.array(P)
lamA=np.array([1.0,100.0]); c,s=np.cos(np.pi/4),np.sin(np.pi/4); Q=np.array([[c,-s],[s,c]])
grad_aligned=lambda t: lamA*t; grad_rotated=lambda t: Q@(lamA*(Q.T@t))
runs=[("plain",0.018,"0.4","plain GD, $\\gamma=0.018$"),("momentum",0.018,"C0","momentum, $\\gamma=0.018$"),
      ("adagrad",0.2,"C1","AdaGrad, $\\gamma=0.2$"),("rmsprop",0.05,"C2","RMSProp, $\\gamma=0.05$"),("adam",0.1,"crimson","Adam, $\\gamma=0.1$")]
fig,axes=plt.subplots(1,2,figsize=(10.0,3.8))
print("  iterations to ||theta||<1e-4 on the kappa=100 bowl (aligned / rotated):")
for ax,grad,title,x0 in [(axes[0],grad_aligned,"axis-aligned: $\\boldsymbol{H}=\\mathrm{diag}(1,100)$",np.array([2.0,1.0])),
                         (axes[1],grad_rotated,"rotated by $45^\\circ$: $\\boldsymbol{H}=\\boldsymbol{Q}\\,\\mathrm{diag}(1,100)\\,\\boldsymbol{Q}^T$",Q@np.array([2.0,1.0]))]:
    g1,g2=np.meshgrid(np.linspace(-1.2,2.4,240),np.linspace(-1.0,2.4,240))
    pts=np.stack([g1.ravel(),g2.ravel()]); H=np.diag(lamA) if grad is grad_aligned else Q@np.diag(lamA)@Q.T
    Z=0.5*np.einsum("ij,ik,kj->j",pts,H,pts).reshape(g1.shape)
    ax.contour(g1,g2,Z,levels=np.geomspace(0.01,60,16),colors="0.75",linewidths=0.7)
    for meth,gamma,col,lab in runs:
        P=optimise(grad,x0,meth,gamma,5000); d=np.linalg.norm(P,axis=1); k=int(np.argmax(d<1e-4)) if (d<1e-4).any() else None
        print(f"    {title[:12]:12s} {meth:8s}: {k}")
        ax.plot(P[:101,0],P[:101,1],"o-",ms=1.8,lw=0.9,color=col,label=lab)
    ax.plot(0,0,"k*",ms=9); ax.set_title(title,fontsize=9.5); ax.set_xlabel(r"$\theta_1$"); ax.set_aspect("equal")
axes[0].set_ylabel(r"$\theta_2$"); axes[0].legend(fontsize=7.5,loc="upper left")
save(fig,4,"adaptive_paths")

# 7. stochastic gradient descent against gradient descent on a large least-squares problem, per flop
n_big,p_big=100_000,20; r2=np.random.default_rng(2026)
z=r2.normal(size=(n_big,p_big)); rho=0.9
xcorr=np.empty_like(z); xcorr[:,0]=z[:,0]
for j in range(1,p_big): xcorr[:,j]=rho*xcorr[:,j-1]+np.sqrt(1-rho**2)*z[:,j]     # AR(1)-correlated columns
xcorr=(xcorr-xcorr.mean(0))/xcorr.std(0)
theta_true=r2.normal(size=p_big); sigma=1.0
ybig=xcorr@theta_true+sigma*r2.normal(size=n_big)
Hbig=(2.0/n_big)*xcorr.T@xcorr; ev=np.linalg.eigvalsh(Hbig); kappa_big=ev.max()/ev.min()
theta_hat=np.linalg.solve(xcorr.T@xcorr,xcorr.T@ybig); c_hat=np.mean((xcorr@theta_hat-ybig)**2)
floor=sigma**2*p_big/n_big
def excess(th): return np.mean((xcorr@th-ybig)**2)-c_hat
flop_per_point=4*p_big                                   # two matrix-vector products per gradient
print(f"  large problem: n={n_big}, p={p_big}, kappa={kappa_big:.0f}, sigma^2 p/n={floor:.1e}, excess cost of theta_true={excess(theta_true):.2e}")
# gradient descent at gamma* for 300 iterations
gstar=2/(ev.max()+ev.min()); th=np.zeros(p_big); gd_fl,gd_ex=[0],[excess(th)]
for k in range(1,1001):
    th=th-gstar*(2.0/n_big)*xcorr.T@(xcorr@th-ybig); gd_fl.append(k*n_big*flop_per_point); gd_ex.append(excess(th))
bstar=((np.sqrt(kappa_big)-1)/(np.sqrt(kappa_big)+1))**2; gmom=4/(np.sqrt(ev.max())+np.sqrt(ev.min()))**2
th=np.zeros(p_big); v=np.zeros(p_big); mo_ex=[excess(th)]
for k in range(1,1001):
    v=bstar*v+gmom*(2.0/n_big)*xcorr.T@(xcorr@th-ybig); th=th-v; mo_ex.append(excess(th))
# SGD, M=32, constant gamma and the schedule of Eq. (4.35), 5 epochs, recorded 20 times per epoch
def sgd_curve(gamma=None,schedule=None,M=32,epochs=5,seed=1):
    r=np.random.default_rng(seed); th=np.zeros(p_big); t=0; fl,ex=[0],[excess(th)]; per=n_big//M
    for e in range(epochs):
        idx=r.permutation(n_big)
        for i,b in enumerate([idx[k:k+M] for k in range(0,n_big,M)]):
            t+=1; g=(2.0/len(b))*xcorr[b].T@(xcorr[b]@th-ybig[b])
            gam=gamma if schedule is None else schedule[0]/(t+schedule[1])
            th=th-gam*g
            if (i+1)%(per//20)==0: fl.append(t*M*flop_per_point); ex.append(excess(th))
    return np.array(fl),np.array(ex)
fig,ax=plt.subplots(figsize=(6.2,3.8))
ax.loglog(np.array(gd_fl[1:]),np.maximum(gd_ex[1:],1e-16),color="0.3",lw=1.8,label=rf"gradient descent, $\gamma^*$, $\kappa={kappa_big:.0f}$")
ax.loglog(np.array(gd_fl[1:]),np.maximum(mo_ex[1:],1e-16),color="0.3",lw=1.4,ls="--",label=rf"momentum, tuned $\gamma,\beta$")
for kw,col,lab in [(dict(gamma=0.02),"C0",r"SGD, $M=32$, $\gamma=0.02$"),(dict(gamma=0.005),"C1",r"SGD, $M=32$, $\gamma=0.005$"),
                   (dict(schedule=(20.0,1000.0)),"crimson",r"SGD, $M=32$, $\gamma_t=20/(t+1000)$")]:
    fl,ex=sgd_curve(**kw); ax.loglog(fl[1:],np.maximum(ex[1:],1e-16),color=col,lw=1.5,label=lab)
    print(f"    SGD {kw}: excess after 1 epoch {ex[20]:.2e}, after 5 epochs {ex[-1]:.2e}")
print(f"    GD: excess after 1 iteration {gd_ex[1]:.2e}, after 10 {gd_ex[10]:.2e}, after 100 {gd_ex[100]:.2e}, after 1000 {gd_ex[1000]:.2e}")
print(f"    momentum: after 10 {mo_ex[10]:.2e}, after 100 {mo_ex[100]:.2e}, after 1000 {mo_ex[1000]:.2e}; first below floor: GD {next((k for k,e in enumerate(gd_ex) if e<floor),None)}, momentum {next((k for k,e in enumerate(mo_ex) if e<floor),None)}")
ax.axhline(floor,color="k",ls=":",lw=1); ax.text(gd_fl[1]*1.1,floor*1.3,r"estimation error $\sigma^2 p/n$",fontsize=8)
for e in (1,5): ax.axvline(e*n_big*flop_per_point,color="0.8",ls="--",lw=0.8)
ax.text(n_big*flop_per_point*1.05,2e-7,"1 epoch",fontsize=7.5,color="0.4"); ax.text(5*n_big*flop_per_point*1.05,2e-7,"5",fontsize=7.5,color="0.4")
ax.set_xlabel("floating-point operations ($4p$ per data point per gradient)"); ax.set_ylabel(r"$C(\boldsymbol{\theta})-C(\hat{\boldsymbol{\theta}})$")
ax.set_ylim(1e-7,30); ax.legend(fontsize=8,loc="upper right")
save(fig,4,"sgd_vs_gd")
