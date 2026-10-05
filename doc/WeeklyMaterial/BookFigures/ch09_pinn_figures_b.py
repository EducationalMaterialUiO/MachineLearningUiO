import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter09_differential_equations"))
from common import plt, np, save
import autograd.numpy as anp
from autograd import grad
from autograd.misc import flatten
from nn_de import network, init_parameters, d_dxk
import pinn_inverse as I

def run_track(D0, seed=3, n_obs=40, noise=0.01, n_iter=3000):
    rng=anp.random.default_rng(seed)
    X_obs=anp.column_stack([rng.uniform(0,1,n_obs), rng.uniform(0,1,n_obs)])
    y_obs=I.exact(X_obs)+rng.normal(0,noise,n_obs)
    def cost(P):
        Dv=P[1][0]
        return anp.mean((I.u_t(P,I.X_col)-Dv*I.u_xx(P,I.X_col))**2) \
             + 10.0*anp.mean((I.u_net(P,X_obs)-y_obs)**2)
    P=[init_parameters([2,30,30,1],"tanh",anp.random.default_rng(1)), anp.array([D0])]
    flat,unflat=flatten(P); g=grad(lambda f: cost(unflat(f)))
    m=np.zeros_like(flat); v=np.zeros_like(flat); track=[]
    for it in range(1,n_iter+1):
        gg=g(flat); m=0.9*m+0.1*gg; v=0.999*v+0.001*gg**2
        flat=flat-1e-2*(m/(1-0.9**it))/(np.sqrt(v/(1-0.999**it))+1e-8)
        if it%20==0 or it==1: track.append((it, float(unflat(flat)[1][0])))
    return track, unflat(flat), X_obs, y_obs

fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for D0,col in [(2.0,"C0"),(1.0,"C1"),(0.1,"C2")]:
    tr,P,X_obs,y_obs=run_track(D0)
    axs[0].plot([t[0] for t in tr],[t[1] for t in tr],col,lw=1.4,
                label=rf"$D_0={D0}$ $\to$ {tr[-1][1]:.3f}")
    if D0==2.0: Pbest,Xo,yo=P,X_obs,y_obs
axs[0].axhline(0.5,color="k",ls="--",lw=1.0,label=r"$D_{\rm true}=0.5$")
axs[0].set_xlabel("iteration"); axs[0].set_ylabel(r"$D$")
axs[0].set_title("(a) the unknown coefficient is recovered"); axs[0].legend(fontsize=8.5)

xline=anp.linspace(0,1,200)
for t,col in [(0.05,"C0"),(0.2,"C1"),(0.5,"C2")]:
    X=anp.column_stack([xline,t*anp.ones(200)])
    axs[1].plot(xline,I.u_net(Pbest,X),col,lw=1.6,label=f"$t={t}$")
    axs[1].plot(xline,I.exact(X),col,ls="--",lw=1.0)
sc=axs[1].scatter(Xo[:,0],yo,c=Xo[:,1],cmap="viridis",s=22,edgecolors="k",
                  linewidths=0.4,zorder=5,label="noisy data")
cb=plt.colorbar(sc,ax=axs[1]); cb.set_label("$t$ of observation",fontsize=9)
axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$u(x,t)$")
axs[1].set_title("(b) solution and the 40 observations"); axs[1].legend(fontsize=8.5)
save(fig,9,"pinn_inverse")
