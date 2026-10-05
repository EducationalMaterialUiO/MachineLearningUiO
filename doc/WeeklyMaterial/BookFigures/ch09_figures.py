from common import *
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/kind-inspiring-knuth/mnt/outputs/ch9")
import numpy as _np
import autograd.numpy as anp
from nn_de import *

# 1. ODE solutions vs exact
fig,axes=plt.subplots(1,3,figsize=(11.0,3.2))
# exponential decay
kappa,g0=2.0,10.0; X=_np.linspace(0,1,50).reshape(-1,1)
def r1(P,Xa):
    N,dN=network_derivs(P,Xa,"tanh",order=1); x=Xa[:,0]
    return (N+x*dN)+kappa*(g0+x*N)
P,_=solve_de(r1,[1,40,40,1],X,"tanh",n_iter=3000,gamma=2e-2,rng=_np.random.default_rng(1))
N,_=network_derivs(P,X,"tanh",order=1); g=g0+X[:,0]*N
axes[0].plot(X[:,0],g0*_np.exp(-kappa*X[:,0]),"k-",lw=2,label="exact")
axes[0].plot(X[:,0],g,"o",ms=3,label="network")
axes[0].set_title(r"$g'=-\kappa g$, $g(0)=10$",fontsize=10)
# logistic
alpha,A,g0b=2.0,1.0,1.2; T=_np.linspace(0,1,50).reshape(-1,1)
def r2(P,Xa):
    N,dN=network_derivs(P,Xa,"tanh",order=1); t=Xa[:,0]
    gt=g0b+t*N; return (N+t*dN)-alpha*gt*(A-gt)
P2,_=solve_de(r2,[1,40,40,1],T,"tanh",n_iter=3000,gamma=2e-2,rng=_np.random.default_rng(1))
N,_=network_derivs(P2,T,"tanh",order=1); g2=g0b+T[:,0]*N
ex2=A*g0b/(g0b+(A-g0b)*_np.exp(-alpha*A*T[:,0]))
tE=_np.linspace(0,1,11); dt=tE[1]-tE[0]; gE=_np.zeros(11); gE[0]=g0b
for i in range(10): gE[i+1]=gE[i]+dt*alpha*gE[i]*(A-gE[i])
axes[1].plot(T[:,0],ex2,"k-",lw=2,label="exact")
axes[1].plot(T[:,0],g2,"o",ms=3,label="network")
axes[1].plot(tE,gE,"s--",ms=4,label="forward Euler, 10 steps")
axes[1].set_title(r"$g'=\alpha g(A-g)$",fontsize=10)
# poisson
f=lambda x:(3*x+x**2)*_np.exp(x); Xp=_np.linspace(0,1,60).reshape(-1,1)
def r3(P,Xa):
    N,dN,d2N=network_derivs(P,Xa,"tanh",order=2); x=Xa[:,0]
    return -(-2*N+2*(1-2*x)*dN+x*(1-x)*d2N)-f(x)
P3,_=solve_de(r3,[1,30,30,1],Xp,"tanh",n_iter=4000,gamma=1e-2,rng=_np.random.default_rng(2))
N,dN,d2N=network_derivs(P3,Xp,"tanh",order=2); g3=Xp[:,0]*(1-Xp[:,0])*N
axes[2].plot(Xp[:,0],Xp[:,0]*(1-Xp[:,0])*_np.exp(Xp[:,0]),"k-",lw=2,label="exact")
axes[2].plot(Xp[:,0],g3,"o",ms=3,label="network")
axes[2].set_title(r"$-g''=f(x)$, $g(0)=g(1)=0$",fontsize=10)
for a in axes: a.set_xlabel("$x$"); a.legend(fontsize=8)
axes[0].set_ylabel("$g$")
save(fig,9,"ode_solutions")

# 2. second derivative by activation
acts=["tanh","gelu","swish","elu","softplus","sigmoid","relu","leaky_relu"]
Xd=_np.linspace(-2,2,60).reshape(-1,1); vals=[]
for act in acts:
    Pa=init_parameters([1,20,20,1],act,_np.random.default_rng(3))
    _,_,d2=network_derivs(Pa,Xd,act); vals.append(_np.abs(d2).max())
fig,ax=plt.subplots(figsize=(6.2,3.4))
cols=["C0"]*6+["crimson"]*2
ax.bar(range(len(acts)),_np.maximum(vals,1e-18),color=cols)
ax.set_yscale("log"); ax.set_xticks(range(len(acts))); ax.set_xticklabels(acts,rotation=30,ha="right")
ax.set_ylabel(r"$\max|\partial^2 N/\partial x^2|$")
ax.set_title("rectified activations give exactly zero",fontsize=10)
ax.axhline(1e-18,color="crimson",ls=":",lw=1)
for i,(v,a_) in enumerate(zip(vals,acts)):
    if v==0: ax.text(i,2e-18,"0",ha="center",fontsize=9,color="crimson")
save(fig,9,"second_derivative_activations")

# 3. PDE solutions
nx,nt=20,20
xs,ts=_np.linspace(0,1,nx),_np.linspace(0,1,nt)
Xg,Tg=_np.meshgrid(xs,ts,indexing="ij"); Xp2=_np.column_stack([Xg.ravel(),Tg.ravel()])
def trial_diff(P,Xa):
    x,t=Xa[:,0],Xa[:,1]
    return (1-t)*anp.sin(anp.pi*x)+x*(1-x)*t*network(P,Xa,"tanh")
ut=d_dxk(trial_diff,1); ux=d_dxk(trial_diff,0); uxx=d_dxk(ux,0)
Pd,_=solve_de(lambda P,Xa: ut(P,Xa)-uxx(P,Xa),[2,30,30,1],Xp2,"tanh",n_iter=800,gamma=1e-2,rng=_np.random.default_rng(1))
u=trial_diff(Pd,Xp2); ex=_np.exp(-_np.pi**2*Xp2[:,1])*_np.sin(_np.pi*Xp2[:,0])
def trial_wave(P,Xa):
    x,t=Xa[:,0],Xa[:,1]
    return (1-t**2)*anp.sin(anp.pi*x)+x*(1-x)*t**2*network(P,Xa,"tanh")
wt=d_dxk(trial_wave,1); wtt=d_dxk(wt,1); wx=d_dxk(trial_wave,0); wxx=d_dxk(wx,0)
Pw,_=solve_de(lambda P,Xa: wtt(P,Xa)-wxx(P,Xa),[2,30,30,1],Xp2,"tanh",n_iter=800,gamma=1e-2,rng=_np.random.default_rng(1))
w=trial_wave(Pw,Xp2); exw=_np.cos(_np.pi*Xp2[:,1])*_np.sin(_np.pi*Xp2[:,0])
fig,axes=plt.subplots(1,3,figsize=(11.0,3.2))
for j,tv in enumerate([0.0,0.05,0.16]):
    it=_np.argmin(abs(ts-tv)); m=_np.isclose(Xp2[:,1],ts[it])
    axes[0].plot(Xp2[m,0],ex[m],"k-",lw=1.6)
    axes[0].plot(Xp2[m,0],u[m],"o",ms=3,label=f"$t={ts[it]:.2f}$")
axes[0].set_title("diffusion: lines exact, dots network",fontsize=10); axes[0].legend(fontsize=8)
axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$g(x,t)$")
c=axes[1].contourf(Xg,Tg,_np.abs(u-ex).reshape(nx,nt),levels=18,cmap="magma")
fig.colorbar(c,ax=axes[1]); axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$t$")
axes[1].set_title("diffusion: absolute error",fontsize=10); axes[1].grid(False)
for tv in [0.0,0.25,0.5,0.75]:
    it=_np.argmin(abs(ts-tv)); m=_np.isclose(Xp2[:,1],ts[it])
    axes[2].plot(Xp2[m,0],exw[m],"k-",lw=1.4)
    axes[2].plot(Xp2[m,0],w[m],"o",ms=3,label=f"$t={ts[it]:.2f}$")
axes[2].set_title("wave: lines exact, dots network",fontsize=10); axes[2].legend(fontsize=7)
axes[2].set_xlabel("$x$")
save(fig,9,"pde_solutions")
