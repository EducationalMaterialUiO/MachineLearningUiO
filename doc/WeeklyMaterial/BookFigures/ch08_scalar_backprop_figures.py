"""Figures for Section 8.8.1 (backpropagation on two scalar networks).

Writes BookFigures/chapter08_neural_networks/scalar_backprop.{pdf,png}.
The hand-written backpropagation of the two scalar networks is checked
against jax.grad at every iteration; the maximum discrepancy is printed.
"""
from common import *
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def sigma(z):  return 1.0/(1.0+np.exp(-z))
def dsigma(z): s = sigma(z); return s*(1.0-s)

# a scalar regression problem: targets in (0,1) around a shifted sigmoid
rng = np.random.default_rng(8)
n = 20
x = np.linspace(-2.0, 2.0, n)
y = sigma(2.0*x - 1.0) + 0.1*rng.standard_normal(n)

# ---- no hidden layer, batch of n samples: cost (1/2n) sum (a1-y)^2 ----
def cost_one(w1, b1):
    return 0.5*np.mean((sigma(w1*x + b1) - y)**2)
def grad_one(w1, b1):
    z1 = w1*x + b1; a1 = sigma(z1)
    delta1 = (a1 - y)*dsigma(z1)/n
    return cost_one(w1, b1), np.sum(delta1*x), np.sum(delta1)

# ---- one hidden layer with one scalar hidden unit -----------------------
def grad_two(w1, b1, w2, b2):
    z1 = w1*x + b1; a1 = sigma(z1)
    z2 = w2*a1 + b2; a2 = sigma(z2)
    delta2 = (a2 - y)*dsigma(z2)/n
    delta1 = delta2*w2*dsigma(z1)
    return (0.5*np.mean((a2-y)**2), np.sum(delta1*x), np.sum(delta1),
            np.sum(delta2*a1), np.sum(delta2))

# the same costs for jax.grad
xj, yj = jnp.array(x), jnp.array(y)
def C1(p): return 0.5*jnp.mean((jax.nn.sigmoid(p[0]*xj + p[1]) - yj)**2)
def C2(p):
    a1 = jax.nn.sigmoid(p[0]*xj + p[1])
    return 0.5*jnp.mean((jax.nn.sigmoid(p[2]*a1 + p[3]) - yj)**2)
g1, g2 = jax.jit(jax.grad(C1)), jax.jit(jax.grad(C2))

gamma, iters = 2.0, 1000
p1 = np.array([0.25, 0.5]); p2 = np.array([0.25, 0.5, 1.0, -0.5])
path1 = [p1.copy()]; cost1 = []; cost2 = []; gw1_one = []; gw1_two = []
err1 = err2 = 0.0
for k in range(iters):
    C, gw, gb = grad_one(*p1)
    err1 = max(err1, np.max(np.abs(np.array([gw, gb]) - np.asarray(g1(jnp.array(p1))))))
    cost1.append(C); gw1_one.append(abs(gw))
    p1 -= gamma*np.array([gw, gb]); path1.append(p1.copy())
    C, gw1, gb1, gw2, gb2 = grad_two(*p2)
    err2 = max(err2, np.max(np.abs(np.array([gw1, gb1, gw2, gb2]) - np.asarray(g2(jnp.array(p2))))))
    cost2.append(C); gw1_two.append(abs(gw1))
    p2 -= gamma*np.array([gw1, gb1, gw2, gb2])
path1 = np.array(path1)
print(f"final: no hidden layer (w1,b1) = {p1.round(3)}, cost {cost1[-1]:.5f}")
print(f"       one hidden layer (w1,b1,w2,b2) = {p2.round(3)}, cost {cost2[-1]:.5f}")
print(f"max |backprop - jax.grad|: {err1:.1e} (no hidden), {err2:.1e} (one hidden)")

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
# (a) cost landscape of the single neuron with the gradient-descent path
W, B = np.meshgrid(np.linspace(-1.0, 4.0, 161), np.linspace(-3.0, 2.0, 161))
Cgrid = np.array([[cost_one(w, b) for w in W[0]] for b in B[:, 0]])
cs = axes[0].contour(W, B, Cgrid, levels=np.geomspace(0.006, Cgrid.max(), 12),
                     colors="gray", linewidths=0.8)
axes[0].plot(path1[:, 0], path1[:, 1], "-", c="darkred", lw=1.4)
axes[0].plot(path1[0, 0], path1[0, 1], "o", c="darkred", ms=6, label="start $(0.25,\\,0.5)$")
axes[0].plot(path1[-1, 0], path1[-1, 1], "s", c="darkred", ms=6, label=f"after {iters} steps")
axes[0].plot(2.0, -1.0, "+", c="darkblue", ms=10, mew=2, label="$(2,-1)$ of the target")
axes[0].set_xlabel("$w^1$"); axes[0].set_ylabel("$b^1$")
axes[0].set_title("no hidden layer: $C(w^1,b^1)$ and the descent", fontsize=10)
axes[0].legend(fontsize=7, loc="lower right")
# (b) training curves
axes[1].semilogy(cost1, c="darkred", label="no hidden layer")
axes[1].semilogy(cost2, c="darkblue", label="one hidden layer")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("$C$")
axes[1].set_title("cost, plain gradient descent, $\\gamma=2$", fontsize=10)
axes[1].legend(fontsize=8)
# (c) the first-layer gradient in the two networks
axes[2].semilogy(gw1_one, c="darkred", label="no hidden layer: $\\delta^1 x$")
axes[2].semilogy(gw1_two, c="darkblue", label="one hidden layer: $\\delta^2 w^2\\sigma'(z^1)\\,x$")
axes[2].set_xlabel("iteration"); axes[2].set_ylabel("$|\\partial C/\\partial w^1|$")
axes[2].set_title("gradient reaching the first weight", fontsize=10)
axes[2].legend(fontsize=7)
save(fig, 8, "scalar_backprop")
