import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten
from nn_de import network, init_parameters, d_dxk

def adam_general(cost, params, n_iter=2000, eta=1e-2, b1=0.9, b2=0.999, eps=1e-8,
                 every=500, verbose=False):
    """Adam on any nested list of arrays -- here the weights plus the unknown D."""
    flat, unflatten = flatten(params)
    gfun = grad(lambda f: cost(unflatten(f)))
    m = np.zeros_like(flat); v = np.zeros_like(flat)
    for it in range(1, n_iter + 1):
        g = gfun(flat)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - eta * (m / (1 - b1**it)) / (np.sqrt(v / (1 - b2**it)) + eps)
        if verbose and (it % every == 0 or it == 1):
            print(f"  it {it:5d}  cost {cost(unflatten(flat)):.4e}  "
                  f"D {unflatten(flat)[1][0]:.6f}")
    return unflatten(flat)

D_true = 0.5
def exact(X, D=D_true): return np.exp(-D*np.pi**2*X[:,1])*np.sin(np.pi*X[:,0])

def u_net(P, X): return network(P[0], X, "tanh")
u_t=d_dxk(u_net,1); u_x=d_dxk(u_net,0); u_xx=d_dxk(u_x,0)

nx=nt=20
xs=np.linspace(0,1,nx); ts=np.linspace(0,1,nt)
Xi,Ti=np.meshgrid(xs[1:-1],ts[1:-1],indexing="ij")
X_col=np.column_stack([Xi.ravel(),Ti.ravel()])

def run(n_obs=40, noise=0.01, seed=3, n_iter=3000, D0=2.0, verbose=False):
    rng=np.random.default_rng(seed)
    X_obs=np.column_stack([rng.uniform(0,1,n_obs), rng.uniform(0,1,n_obs)])
    y_obs=exact(X_obs)+rng.normal(0,noise,n_obs)
    def cost(P):
        D=P[1][0]
        r_pde=u_t(P,X_col)-D*u_xx(P,X_col)
        r_dat=u_net(P,X_obs)-y_obs
        return np.mean(r_pde**2)+10.0*np.mean(r_dat**2)
    P=[init_parameters([2,30,30,1],"tanh",np.random.default_rng(1)), np.array([D0])]
    P=adam_general(cost,P,n_iter=n_iter,eta=1e-2,verbose=verbose)
    return P[1][0], X_obs, y_obs

if __name__=="__main__":
    D,_,_=run(verbose=True)
    print(f"\nD_true = {D_true}, recovered D = {D:.6f}, rel err {abs(D-D_true)/D_true:.2%}")
