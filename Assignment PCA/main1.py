import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Load the data from sklearn module
df = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
df['MEDV'] = pd.DataFrame(load_boston().target)
print('Shape of Data is : {} rows and {} columns'.format(df.shape[0], df.shape[1]))


pca = PCA(n_components=13)
X = df.drop('MEDV', axis=1)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca,
                      columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10',
                                   'PCA11', 'PCA12', 'PCA13'])
df_pca['MEDV'] = df['MEDV']

# Lets look at the correlation matrix now.
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
sns.heatmap(df_pca.corr(), annot=True)

# Lets look at the distribution of our features after applying PCA
pos = 1
fig = plt.figure(figsize=(16, 24))
for i in df_pca.columns:
    ax = fig.add_subplot(7, 2, pos)
    pos = pos + 1
    sns.distplot(df_pca[i], ax=ax)
plt.show()

