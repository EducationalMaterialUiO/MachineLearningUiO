"""Chapter 2: listing 8, from the section on markov chains and the metropolis algorit.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np

def metropolis(log_p, x0, n_steps, step_size=1.0, rng=None):
    """Sample from a distribution known up to a constant.

    log_p returns the logarithm of the unnormalised density, which is the
    numerically sensible way to handle the ratio in Eq. (2.metropolis).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    chain = np.empty((n_steps, x.size))
    logp_x = log_p(x)
    accepted = 0

    for t in range(n_steps):
        proposal = x + step_size * rng.normal(size=x.size)   # symmetric proposal
        logp_new = log_p(proposal)
        # accept if log of the ratio exceeds the log of a uniform deviate
        if np.log(rng.random()) < logp_new - logp_x:
            x, logp_x = proposal, logp_new
            accepted += 1
        chain[t] = x

    return chain, accepted / n_steps


rng = np.random.default_rng(2024)
# Unnormalised standard Gaussian: the constant 1/sqrt(2 pi) is never needed
chain, rate = metropolis(lambda x: -0.5 * np.sum(x**2), x0=[0.0],
                         n_steps=100000, step_size=2.0, rng=rng)

burn_in = 1000
samples = chain[burn_in:, 0]
print(f"acceptance rate {rate:.3f}")
print(f"mean {samples.mean():.4f}  variance {samples.var():.4f}")
