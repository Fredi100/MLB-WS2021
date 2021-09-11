import math


class KFold:

    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.fold_size = math.floor(data.shape[0] / k)

    def get_folds(self, i):
        test_start = i * self.fold_size
        test_end = test_start + self.fold_size  # exclusive end index

        train = self.data.drop(range(test_start, test_end))
        test = self.data.iloc[test_start:test_end]

        X_train = train.iloc[:, :-1]
        X_test = test.iloc[:, :-1]
        y_train = train.iloc[:, -1]
        y_test = test.iloc[:, -1]

        return [X_train, X_test, y_train, y_test]
