"""Chapter 5: listing 13, from the section on the wisconsin breast cancer data.

Extracted from doc/BookML/chapter5.tex.
"""

import matplotlib.pyplot as plt
import pandas as pd

cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()
for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, alpha=0.5)
    ax[i].hist(benign[:, i], bins=bins, alpha=0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc="best")
fig.tight_layout()
plt.show()

# The correlation matrix of Section 1.covariance, computed with pandas
correlations = cancerpd.corr()
plt.figure(figsize=(10, 9))
plt.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(); plt.title("Correlation matrix of the features")
plt.show()
