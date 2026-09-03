"""Chapter 3: listing 14, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

for name, fit in (("Ridge", ridge_fit), ("Lasso", lasso_fit)):
    selected = [select(lambdas, *cv_curve(fit, X_train, y_train, lambdas, K=5,
                                          rng=np.random.default_rng(seed)))[0]
                for seed in range(20)]
    print(f"{name}: lambda_min over 20 shuffles between"
          f" {min(selected):.4f} and {max(selected):.4f}")
