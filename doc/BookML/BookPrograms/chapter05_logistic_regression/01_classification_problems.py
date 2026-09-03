"""Chapter 5: listing 1, from the section on classification problems.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chd = pd.read_csv("DataFiles/chddata.csv", names=("ID", "Age", "Agegroup", "CHD"))
plt.scatter(chd["Age"], chd["CHD"], marker="o")
plt.axis([18, 70.0, -0.1, 1.2])
plt.xlabel("Age"); plt.ylabel("CHD")
plt.show()
