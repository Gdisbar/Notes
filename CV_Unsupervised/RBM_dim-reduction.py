import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, _, _ = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Set up RBM and PCA for dimensionality reduction
rbm = BernoulliRBM(n_components=2, learning_rate=0.01, n_iter=20, random_state=42)
pca = PCA(n_components=2)

# Create a pipeline for RBM and PCA
rbm_pca = Pipeline(steps=[('rbm', rbm), ('pca', pca)])

# Fit and transform the data using the pipeline
X_rbm_pca = rbm_pca.fit_transform(X_train)

# Plot the reduced-dimensional data
plt.figure(figsize=(12, 5))

# Plot original data
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot RBM + PCA transformed data
plt.subplot(1, 2, 2)
plt.scatter(X_rbm_pca[:, 0], X_rbm_pca[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.title('RBM + PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
