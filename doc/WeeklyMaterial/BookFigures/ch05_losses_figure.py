"""Chapter 5, section on other loss functions: the margin losses (left) and the
probabilities each of them estimates when minimised with a linear model on
data from a true logistic model (right).  Writes margin_losses.{pdf,png}."""
from common import *
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit
from jax.nn import sigmoid, softplus, log_sigmoid

def cross_entropy(t, y): return softplus(t) - y * t
def squared_error(t, y): return (y - sigmoid(t))**2
def hinge(t, y):         return jnp.maximum(0.0, 1.0 - (2*y-1)*t)
def squared_hinge(t, y): return jnp.maximum(0.0, 1.0 - (2*y-1)*t)**2
def exponential(t, y):   return jnp.exp(-(2*y-1)*t)
def focal(t, y, gamma=2.0):
    p = sigmoid(t)
    return -(y*(1-p)**gamma*log_sigmoid(t) + (1-y)*p**gamma*log_sigmoid(-t))
LOSSES = [("cross entropy", cross_entropy), ("squared error", squared_error),
          ("hinge", hinge), ("squared hinge", squared_hinge),
          ("exponential", exponential), (r"focal, $\gamma=2$", focal)]

def cost(theta, X, y, loss): return jnp.mean(loss(X @ theta, y))
def fit(loss, X, y, eta=0.5, epochs=3000):
    step = jit(lambda th: th - eta * grad(cost)(th, X, y, loss))
    th = jnp.zeros(X.shape[1])
    for _ in range(epochs): th = step(th)
    return th

rng = np.random.default_rng(2024)
n = 400
X = jnp.asarray(np.c_[np.ones(n), rng.normal(size=(n, 2))])
theta_true = jnp.array([0.5, 2.0, -1.0])
p_true = sigmoid(X @ theta_true)
y = jnp.asarray((rng.random(n) < np.asarray(p_true)).astype(float))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
m = np.linspace(-3, 3, 400)
for k, (name, loss) in enumerate(LOSSES):
    ax1.plot(m, np.asarray(loss(jnp.asarray(m), 1.0)), label=name, color=f"C{k}", lw=1.6)
ax1.step([-3, 0, 3], [1, 1, 0], where="post", color="k", lw=1.2, ls=":", label="0-1 loss")
ax1.set_ylim(0, 4); ax1.set_xlabel(r"margin $m=\tilde y\,\mathbf{x}^T\boldsymbol{\theta}$")
ax1.set_ylabel(r"loss $\ell(m)$"); ax1.legend(fontsize=8)
order = np.argsort(np.asarray(p_true)); pt = np.asarray(p_true)[order]
ax2.plot([0, 1], [0, 1], "k:", lw=1.2, label="calibrated")
for k, (name, loss) in enumerate(LOSSES):
    th = fit(loss, X, y)
    ax2.plot(pt, np.asarray(sigmoid(X @ th))[order], color=f"C{k}", lw=1.6, label=name)
ax2.set_xlabel(r"true probability $p(y=1\mid \mathbf{x})$")
ax2.set_ylabel(r"fitted $\sigma(\mathbf{x}^T\hat{\boldsymbol{\theta}})$"); ax2.legend(fontsize=8)
save(fig, 5, "margin_losses")
