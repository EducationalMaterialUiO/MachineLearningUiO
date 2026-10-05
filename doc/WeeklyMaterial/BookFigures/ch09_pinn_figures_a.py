import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter09_differential_equations"))
from common import plt, np, save
import autograd.numpy as anp
import diff_pinn as D
from pinn import pinn_solve
from nn_de import network, d_dxk, solve_de
from autograd.misc import flatten
from autograd import grad
from nn_de import init_parameters

# ---------- PINN diffusion with recorded history ----------
terms=[("pde",1.0,D.r_pde,D.X_col),("ic",10.0,D.r_ic,D.X_ic),
       ("bcL",10.0,D.r_bc,D.X_l),("bcR",10.0,D.r_bc,D.X_r)]
P_pinn,hist=pinn_solve(terms,[2,30,30,1],"tanh",n_iter=4000,gamma=1e-2,
                       rng=anp.random.default_rng(1),every=50)
# ---------- hard-constraint reference ----------
def trial(P,X):
    x,t=X[:,0],X[:,1]
    return (1-t)*anp.sin(anp.pi*x)+x*(1-x)*t*network(P,X,"tanh")
ut=d_dxk(trial,1); ux=d_dxk(trial,0); uxx=d_dxk(ux,0)
xs=anp.linspace(0,1,20); Xg,Tg=anp.meshgrid(xs,xs,indexing="ij")
Xall=anp.column_stack([Xg.ravel(),Tg.ravel()])
P_hard,_=solve_de(lambda P,X: ut(P,X)-uxx(P,X),[2,30,30,1],Xall,"tanh",
                  n_iter=4000,gamma=1e-2,rng=anp.random.default_rng(1))

fig,axs=plt.subplots(1,3,figsize=(13,3.8))
its=[h[0] for h in hist]
for key,lab,col in [("pde",r"$L_{\rm PDE}$","C0"),("ic",r"$L_{\rm IC}$","C1"),
                    ("bcL",r"$L_{\rm BC},\ x=0$","C2"),("bcR",r"$L_{\rm BC},\ x=1$","C3")]:
    axs[0].plot(its,[h[2][key] for h in hist],col,label=lab,lw=1.3)
axs[0].plot(its,[h[1] for h in hist],"k--",lw=1.0,label="total")
axs[0].set_yscale("log"); axs[0].set_xlabel("iteration"); axs[0].set_ylabel("loss term")
axs[0].set_title("(a) the loss splits into competing parts"); axs[0].legend(fontsize=8,ncol=2)

n=100; xf=anp.linspace(0,1,n); tf=anp.linspace(0,1,n)
Xf,Tf=anp.meshgrid(xf,tf,indexing="ij"); Xe=anp.column_stack([Xf.ravel(),Tf.ravel()])
err=np.abs(D.u_net(P_pinn,Xe)-D.exact(Xe)).reshape(n,n)
c=axs[1].contourf(Xf,Tf,err,levels=20,cmap="cividis")
plt.colorbar(c,ax=axs[1]); axs[1].grid(False)
axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$t$")
axs[1].set_title("(b) PINN absolute error")

xline=anp.linspace(0,1,200); X0=anp.column_stack([xline,anp.zeros(200)])
axs[2].semilogy(xline,np.abs(D.u_net(P_pinn,X0)-anp.sin(anp.pi*xline))+1e-18,
                "C3",label="soft (PINN)")
hard0=np.abs(trial(P_hard,X0)-anp.sin(anp.pi*xline))
axs[2].semilogy(xline,hard0+1e-18,"C0",label="hard (trial solution)")
axs[2].set_ylim(1e-19,1e-1); axs[2].set_xlabel("$x$")
axs[2].set_ylabel(r"$|u-\sin\pi x|$ at $t=0$")
axs[2].annotate("identically zero,\nfor every $P$", xy=(0.5,1e-18), xytext=(0.30,1e-14),
                fontsize=8.5, color="C0", ha="center",
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.9))
axs[2].set_title("(c) the initial condition at $t=0$"); axs[2].legend(fontsize=9, loc="upper right")
save(fig,9,"pinn_diffusion")
print("hard t=0 max:",hard0.max())
