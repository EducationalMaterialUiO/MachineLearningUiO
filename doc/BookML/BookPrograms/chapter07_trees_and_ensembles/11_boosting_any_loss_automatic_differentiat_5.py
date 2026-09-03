"""Chapter 7: listing 11, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

rng = np.random.default_rng(1)
Xr, yr = make_regression(n_samples=400, n_features=5, noise=10.0, random_state=1)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, random_state=1)
yr_dirty = yr_train.copy()
bad = rng.choice(len(yr_train), 20, replace=False)
yr_dirty[bad] += rng.choice([-1.0, 1.0], 20) * 500.0            # twenty gross outliers

for name, loss in [("squared error", squared),
                   ("Huber, delta=20", lambda y, f: huber(y, f, 20.0))]:
    for label, target in [("clean   ", yr_train), ("outliers", yr_dirty)]:
        model = Boosting(loss, order=1, n_estimators=300, learning_rate=0.1,
                         max_depth=3).fit(Xr_train, target)
        mse = np.mean((model.decision_function(Xr_test) - yr_test)**2)
        print(f"{name:16s} trained on {label} data: test MSE = {mse:7.1f}")

for tau in [0.1, 0.5, 0.9]:
    model = Boosting(lambda y, f: quantile(y, f, tau), order=1, n_estimators=300,
                     learning_rate=0.1, max_depth=3).fit(Xr_train, yr_train)
    below = np.mean(yr_test < model.decision_function(Xr_test))
    print(f"quantile loss, tau = {tau}: fraction of test targets below = {below:.3f}")
