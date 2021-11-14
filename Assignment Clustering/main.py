import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
import random
from util import nearest_cluster, plot_clusters
from scipy.spatial import distance


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid    # only features
        self.members = []

    def update_centroid(self):
        if len(self.members) == 0:
            return
        self.centroid = np.mean(self.members, axis=0)
        return self.centroid

    def clear_members(self):
        self.members = []


def k_means(data, k, max_iter):
    # initialize k random clusters
    clusters = []
    for rand_i in random.sample(range(0, k), k):
        rand_sample = data.loc[rand_i]
        cluster = Cluster(rand_sample)          # initialize with centroid

        clusters.append(cluster)

    # do for amount of iterations
    for i in range(0, max_iter):
        # classify each sample by finding nearest cluster by centroid
        for _, sample in data.iterrows():
            nearest_i = nearest_cluster(sample, clusters)
            clusters[nearest_i].members.append(sample)

        plot_clusters(clusters)

        # check for convergence while also updating centroids
        if all(list(distance.euclidean(cluster.centroid, cluster.update_centroid(), 2) < 0.1 for cluster in clusters)):
            print('breaking due to convergence')
            break

        # clear members for next iteration
        map(lambda x: x.clear_members(), clusters)


if __name__ == '__main__':
    X, _ = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=2, n_samples=100)
    data = pd.DataFrame(data=X, columns=['X1', 'X2'])

    k = 3
    max_iter = 10
    k_means(data, k, max_iter)