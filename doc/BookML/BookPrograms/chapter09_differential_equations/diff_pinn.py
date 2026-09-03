import autograd.numpy as np
from nn_de import network, d_dxk
from pinn import pinn_solve

def u_net(P, X): return network(P, X, ACT_NAME)
ACT_NAME="tanh"

nx=nt=20
xs=np.linspace(0,1,nx); ts=np.linspace(0,1,nt)
# interior collocation points
Xi,Ti=np.meshgrid(xs[1:-1],ts[1:-1],indexing="ij")
X_col=np.column_stack([Xi.ravel(),Ti.ravel()])
X_ic  =np.column_stack([xs, np.zeros(nx)])
X_l   =np.column_stack([np.zeros(nt), ts])
X_r   =np.column_stack([np.ones(nt),  ts])

u_t  = d_dxk(u_net,1); u_x = d_dxk(u_net,0); u_xx = d_dxk(u_x,0)
def r_pde(P,X): return u_t(P,X)-u_xx(P,X)
def r_ic (P,X): return u_net(P,X)-np.sin(np.pi*X[:,0])
def r_bc (P,X): return u_net(P,X)

def exact(X): return np.exp(-np.pi**2*X[:,1])*np.sin(np.pi*X[:,0])

Xf,Tf=np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100),indexing="ij")
X_eval=np.column_stack([Xf.ravel(),Tf.ravel()])

def report(P,tag):
    p=u_net(P,X_eval); e=exact(X_eval)
    err=np.abs(p-e)
    rmse=np.sqrt(np.mean((p-e)**2))
    # edge errors
    m0=X_eval[:,1]==0.0
    bl=X_eval[:,0]==0.0; br=X_eval[:,0]==1.0
    print(f"{tag:28s} max {err.max():.3e}  rmse {rmse:.3e}  "
          f"t=0 {err[m0].max():.3e}  x=0 {err[bl].max():.3e}  x=1 {err[br].max():.3e}")
    return err.max(), rmse

if __name__=="__main__":
    import sys
    print("collocation:",X_col.shape[0],"ic:",len(X_ic),"bc:",len(X_l)+len(X_r))
    for w in [1.0, 10.0, 100.0]:
        terms=[("pde",1.0,r_pde,X_col),("ic",w,r_ic,X_ic),
               ("bcL",w,r_bc,X_l),("bcR",w,r_bc,X_r)]
        P,h=pinn_solve(terms,[2,30,30,1],"tanh",n_iter=800,eta=1e-2,
                       rng=np.random.default_rng(1))
        report(P,f"PINN w={w:g}, 800 it")
        print("   final parts:",{k:f"{v:.2e}" for k,v in h[-1][2].items()})

def longrun():
    print("\n--- longer training, w=10 ---")
    terms=[("pde",1.0,r_pde,X_col),("ic",10.0,r_ic,X_ic),
           ("bcL",10.0,r_bc,X_l),("bcR",10.0,r_bc,X_r)]
    P,h=pinn_solve(terms,[2,30,30,1],"tanh",n_iter=4000,eta=1e-2,
                   rng=np.random.default_rng(1),every=500,verbose=True)
    report(P,"PINN w=10, 4000 it")
    return P,h
