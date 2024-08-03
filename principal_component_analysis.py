from sklearn.decomposition import PCA
import numpy as np

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Features

# Applying PCA
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("Reduced Data:\n", X_reduced)
