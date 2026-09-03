"""Chapter 5: listing 2, from the section on classification problems.

Extracted from doc/BookML/chapter5.tex.
"""

agegroupmean = np.array([0.1, 0.133, 0.250, 0.333, 0.462, 0.625, 0.765, 0.800])
group = np.array([1, 2, 3, 4, 5, 6, 7, 8])
plt.plot(group, agegroupmean, "r-")
plt.axis([0, 9, 0, 1.0])
plt.xlabel("Age group"); plt.ylabel("CHD mean value")
plt.show()
