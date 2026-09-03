"""Chapter 5: listing 9, from the section on one program for every loss.

Extracted from doc/BookML/chapter5.tex.
"""

def evaluate(theta):
    p = sigmoid(X @ theta)
    accuracy = float(jnp.mean((p > 0.5) == (y > 0.5)))
    calibration = float(jnp.mean(jnp.abs(p - p_true)))    # mean |p_hat - p_true|
    return accuracy, calibration

print(f"{'loss':16s} {'theta':26s} accuracy   mean |p_hat - p_true|")
for name, loss in LOSSES.items():
    theta = fit(loss, X, y)
    acc, cal = evaluate(theta)
    print(f"{name:16s} {np.array2string(np.asarray(theta), precision=2):26s}"
          f" {acc:.3f}      {cal:.3f}")
print("true            ", np.asarray(theta_true))
