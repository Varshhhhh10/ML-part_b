from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

# Hierarchical clustering
Z = linkage(X, 'ward')
dendrogram(Z)
plt.show()
