"""Chapter 9 core solver -- reproduced from the chapter listings."""
import autograd.numpy as np
from autograd import elementwise_grad

def sigmoid(z):   return 1.0/(1.0+np.exp(-z))
def tanh_(z):     return np.tanh(z)
def relu(z):      return np.maximum(0.0, z)
def leaky_relu(z,a=0.01): return np.where(z>0, z, a*z)
def elu(z,a=1.0): return np.where(z>0, z, a*(np.exp(np.minimum(z,0))-1))
def softplus(z):  return np.log(1.0+np.exp(-np.abs(z)))+np.maximum(z,0.0)
def swish(z):     return z*sigmoid(z)
def gelu(z):      return 0.5*z*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(z+0.044715*z**3)))
def mish(z):      return z*np.tanh(softplus(z))
ACT={"sigmoid":sigmoid,"tanh":tanh_,"relu":relu,"leaky_relu":leaky_relu,
     "elu":elu,"softplus":softplus,"swish":swish,"gelu":gelu,"mish":mish}

def init_parameters(layer_sizes, activation="tanh", rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    P=[]
    for i in range(len(layer_sizes)-1):
        nin,nout=layer_sizes[i],layer_sizes[i+1]
        s=np.sqrt(2.0/nin) if activation in ("relu","leaky_relu","elu") else np.sqrt(1.0/nin)
        P.append([rng.normal(0,s,(nin,nout)), np.zeros(nout)])
    return P

def network(P, X, activation="tanh"):
    f=ACT[activation]; a=X
    for l,(W,b) in enumerate(P):
        z=a@W+b
        a=f(z) if l<len(P)-1 else z
    return a[:,0]

def adam_minimise(cost, P, n_iter=2000, eta=1e-2, b1=0.9, b2=0.999, eps=1e-8,
                  verbose=False, every=200):
    g=elementwise_grad(cost) if False else None
    from autograd import grad as _grad
    gfun=_grad(cost)
    m=[[np.zeros_like(W),np.zeros_like(b)] for W,b in P]
    v=[[np.zeros_like(W),np.zeros_like(b)] for W,b in P]
    hist=[]
    for it in range(1,n_iter+1):
        G=gfun(P)
        for l in range(len(P)):
            for j in range(2):
                m[l][j]=b1*m[l][j]+(1-b1)*G[l][j]
                v[l][j]=b2*v[l][j]+(1-b2)*G[l][j]**2
                mh=m[l][j]/(1-b1**it); vh=v[l][j]/(1-b2**it)
                P[l][j]=P[l][j]-eta*mh/(np.sqrt(vh)+eps)
        if it%every==0 or it==1:
            c=cost(P); hist.append((it,c))
            if verbose: print(f"  it {it:5d}  cost {c:.4e}")
    return P, hist

def solve_de(residual, layer_sizes, X, activation="tanh", n_iter=2000, eta=1e-2, rng=None):
    P = init_parameters(layer_sizes, activation, rng)
    def cost(P): return np.mean(residual(P, X) ** 2)
    return adam_minimise(cost, P, n_iter=n_iter, eta=eta)

def d_dxk(fun, k):
    def wrapped(P, X):
        def scalarised(xk):
            Xn = np.concatenate([X[:, :k], xk.reshape(-1, 1), X[:, k+1:]], axis=1)
            return fun(P, Xn)
        return elementwise_grad(scalarised)(X[:, k])
    return wrapped
