from scipy.spatial import distance


class KNN:

    def __init__(self, k, x_train, y_train, classifier=True):
        self.y_train = y_train
        self.x_train = x_train
        self.k = k
        self.classifier = classifier

    def __calc_distances(self, x_predict):
        distances = []
        # calc distances
        for X, y in zip(self.x_train.values.tolist(), self.y_train.values.tolist()):
            dist = distance.euclidean(X, x_predict)
            distances.append((dist, y))
        # sort ascending
        distances.sort(key=lambda x: x[0])
        return distances

    def __classifier(self, distances):
        """
        Classifier based on knn

        Counts the appearances of all classes in k neighbours and returns the most frequent one
        """
        count = dict.fromkeys(set(self.y_train), 0)

        for i in range(self.k):
            count[distances[i][1]] += 1
        return max(count, key=count.get)

    def __regressor(self, distances):
        """
        Regressor based on knn

        Sums up all k neighbours and then takes the mean of the sum
        """
        sum = 0
        for i in range(self.k):
            sum += distances[i][1]
        return sum / self.k

    def predict(self, x_predict):
        distances = self.__calc_distances(x_predict)

        if self.classifier:
            return self.__classifier(distances)
        else:
            return self.__regressor(distances)
