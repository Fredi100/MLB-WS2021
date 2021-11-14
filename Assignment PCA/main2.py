import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

# import the iris dataset:
# 3 target classes: setosa, versicolor, virginica
# 4 predictive features: sepal length, sepal width, petal length, petal width
# 150 samples
iris = datasets.load_boston()
X = iris.data
y = iris.target
fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
nrComponents = 13
pca = decomposition.PCA(n_components=nrComponents, svd_solver='full')  # set up the pca class
pca.fit(X)  # fit the data
X = pca.transform(X)  # apply dimensionality reduction: X is projected on the principal components

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.jet)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()

#scree plot
x = np.arange(1, nrComponents + 1)  # 1 to 13 for components
print(pca.explained_variance_ratio_)  # how much the eigenvalues cover
plt.figure()
plt.bar(x, pca.explained_variance_ratio_)
plt.xlabel("principal components")
plt.ylabel("explained variance ratio")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 9),
                         sharey=True, sharex=True)


fs = 9
axes[0].bar(x, pca.components_[0])  # loadings for PC1
axes[0].set_title("loadings (components) of PC1", fontsize=fs)
axes[1].bar(x, pca.components_[1])
axes[1].set_title("loadings (components) of PC2", fontsize=fs)
axes[2].bar(x, pca.components_[2])
axes[2].set_title("loadings (components) of PC3", fontsize=fs)
axes[3].bar(x, pca.components_[3])
axes[3].set_title("loadings (components) of PC4", fontsize=fs)
axes[3].set_xticks(x)
axes[3].set_xticklabels(iris.feature_names)
plt.show()