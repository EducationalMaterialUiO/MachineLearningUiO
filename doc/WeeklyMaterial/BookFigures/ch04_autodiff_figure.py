"""Chapter 4, Section on automatic differentiation: the error of the central
finite difference against the step h, compared with the reverse-mode
derivative from JAX.  Writes fd_vs_ad_error.{pdf,png}."""
from common import *
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

def f(x): return np.sin(2*np.pi*x + x**2)
x0 = 1.0
exact = (2*np.pi + 2)*np.cos(2*np.pi + 1)
hs = np.logspace(-13, -1, 121)
err = np.array([abs((f(x0+h) - f(x0-h))/(2*h) - exact) for h in hs])
eps = np.finfo(float).eps
f3 = -(2*np.pi+2)**3*np.cos(2*np.pi+1) - 6*(2*np.pi+2)*np.sin(2*np.pi+1)  # f'''(1)
model = hs**2/6*abs(f3) + eps*abs(f(x0))/hs
ad_err = abs(float(grad(lambda x: jnp.sin(2*jnp.pi*x + x**2))(x0)) - exact)
ad_err = max(ad_err, eps)          # zero cannot be drawn on a log axis

fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.loglog(hs, err, "o", ms=3.2, color="C0", label="central difference, measured")
ax.loglog(hs, model, "--", color="C3", lw=1.3,
          label=r"$h^2|f'''|/6+\epsilon_M|f|/h$")
ax.axhline(ad_err, color="C2", lw=1.6, label="reverse-mode AD (JAX)")
ax.axvline(eps**(1/3), color="k", lw=0.8, ls=":")
ax.text(eps**(1/3)*1.4, 1e-12, r"$h=\epsilon_M^{1/3}$", fontsize=9)
ax.set_xlabel(r"step $h$"); ax.set_ylabel(r"$|D_hf(1)-f'(1)|$")
ax.set_ylim(1e-17, 1e1)
ax.legend(loc="upper center", fontsize=9)
save(fig, 4, "fd_vs_ad_error")
print("AD error:", ad_err, "  best FD:", err.min(), "at h =", hs[err.argmin()])
