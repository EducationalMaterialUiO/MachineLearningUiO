import numpy as np

def sigmoid(t):                          # the stable form of Section 5.7
    a = np.exp(-np.abs(t))
    return np.where(t >= 0, 1.0 / (1.0 + a), a / (1.0 + a))

def make_data(n=50_000, p=20, rho=0.9, seed=2026):
    """AR(1)-correlated standardised features, Bernoulli labels."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, p - 1))
    xc = np.empty_like(z); xc[:, 0] = z[:, 0]
    for j in range(1, p - 1):
        xc[:, j] = rho * xc[:, j - 1] + np.sqrt(1 - rho**2) * z[:, j]
    xc = (xc - xc.mean(0)) / xc.std(0)
    X = np.column_stack([np.ones(n), xc])
    theta_true = 0.5 * rng.normal(size=p)
    y = (rng.random(n) < sigmoid(X @ theta_true)).astype(float)
    scales = np.concatenate([[1.0], 10.0**rng.uniform(-1, 1, p - 1)])
    return X, y, theta_true, scales     # scales: the "raw" units

def cost(X, y, theta):                   # mean cross entropy, Eq. (5.costmean)
    z = X @ theta
    return np.mean(np.logaddexp(0.0, z) - y * z)

def gradient(X, y, theta):               # Eq. (5.minibatchgrad), any rows
    return X.T @ (sigmoid(X @ theta) - y) / X.shape[0]

def optimiser_step(method, theta, g, state, t, gamma, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """The step function of Section 4.12, without momentum."""
    if method == "plain":
        return theta - gamma * g
    if method == "adagrad":                                # Eq. (4.adagrad)
        state["r"] = r = state.get("r", 0.0) + g * g
        return theta - gamma * g / (np.sqrt(r) + eps)
    if method == "rmsprop":                                # Eq. (4.rmsprop)
        state["r"] = r = rho * state.get("r", 0.0) + (1 - rho) * g * g
        return theta - gamma * g / (np.sqrt(r) + eps)
    if method == "adam":                                   # Eq. (4.adam)
        state["m"] = m = beta1 * state.get("m", 0.0) + (1 - beta1) * g
        state["r"] = r = beta2 * state.get("r", 0.0) + (1 - beta2) * g * g
        m_hat, r_hat = m / (1 - beta1**t), r / (1 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps)

def train(X, y, method, gamma, M=None, epochs=20, t0=None, seed=1):
    """Minibatch loop, Eq. (5.sgdupdate); M=None is full-batch gradient
    descent.  Returns theta at the end of every epoch."""
    n, p = X.shape
    M = n if M is None else M
    rng = np.random.default_rng(seed)
    theta, state, t, history = np.zeros(p), {}, 0, []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for k in range(0, n, M):                   # one epoch = n/M updates
            t += 1
            gam = gamma if t0 is None else gamma * t0 / (t + t0)
            b = idx[k:k + M]
            g = gradient(X[b], y[b], theta)
            theta = optimiser_step(method, theta, g, state, t, gam)
        history.append(theta.copy())
    return history

def newton(X, y):                        # the reference, to machine precision
    theta = np.zeros(X.shape[1])
    for _ in range(50):
        pr = sigmoid(X @ theta)
        H = X.T @ ((pr * (1 - pr))[:, None] * X) / len(y)
        step = np.linalg.solve(H, X.T @ (pr - y) / len(y))
        theta -= step
        if np.linalg.norm(step) < 1e-12:
            break
    return theta

X, y, theta_true, scales = make_data()
settings = {                             # the winners of the learning-rate grid
    "standardised": [("SGD", "plain", 0.01), ("AdaGrad", "adagrad", 0.1),
                     ("RMSProp", "rmsprop", 0.001), ("Adam", "adam", 0.001)],
    "raw":          [("SGD", "plain", 0.03), ("AdaGrad", "adagrad", 0.3),
                     ("RMSProp", "rmsprop", 0.003), ("Adam", "adam", 0.003)]}
for label, Xd, th_true in (("standardised", X, theta_true),
                           ("raw", X * scales, theta_true / scales)):
    c_star = cost(Xd, y, newton(Xd, y))
    lam_max = np.linalg.eigvalsh(Xd.T @ Xd / len(y)).max()
    gamma_gd = 8.0 / lam_max                       # Eq. (5.gdstability)
    runs = [("GD, 200 epochs",
             dict(method="plain", gamma=gamma_gd, epochs=200))]
    runs += [(name, dict(method=m, gamma=gam, M=32))
             for name, m, gam in settings[label]]
    if label == "standardised":
        runs.insert(2, ("SGD, decaying",
                        dict(method="plain", gamma=0.1, M=32, t0=1e3)))
    print(f"{label} features: C* = {c_star:.5f}, "
          f"excess of theta_true {cost(Xd, y, th_true) - c_star:.1e}")
    print("  method           gamma    excess after 1 / 5 / 20 epochs"
          "   mean 15-20   accuracy")
    for name, kw in runs:
        hist = train(Xd, y, **kw)
        ex = np.array([cost(Xd, y, th) - c_star for th in hist])
        if len(ex) > 20:                       # gradient descent, 200 epochs
            e_last, e_mean = ex[199], ex[199]
        else:                                  # SGD: epoch 20, mean of 15-20
            e_last, e_mean = ex[19], ex[14:].mean()
        acc = np.mean((Xd @ hist[-1] > 0) == y)
        print(f"  {name:16s} {kw['gamma']:<7.3g}  {ex[0]:.1e} / {ex[4]:.1e} /"
              f" {e_last:.1e}      {e_mean:.1e}     {acc:.4f}")
