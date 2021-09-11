
import pandas as pd
from KNN import KNN
from KFold import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import sklearn.datasets as datasets
import numpy

# Load Data
dataset = datasets.load_boston()

data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset.target

k = 3
num_folds = 5
mean_accuracies_myKNN = []
mean_accuracies_sklearn = []

# hyperparam tuning with odd values of k
for k in range(1, 10, 2):
    kFold = KFold(num_folds, data)

    my_fold_errors = []
    sklearn_fold_errors = []

    for i in range(0, num_folds):
        # get split
        x_train, x_test, y_train, y_test = kFold.get_folds(i)

        # fit regressors
        myKNN = KNN(k, x_train, y_train, False)
        sklearnKNN = KNeighborsRegressor(k)
        sklearnKNN.fit(x_train, y_train)

        total = x_test.shape[0]
        correct_myKNN = 0
        correct_sklearn = 0

        my_predictions = []
        sklearn_predictions = []

        for x_predict in x_test.values.tolist():
            my_predictions.append(myKNN.predict(x_predict))
            sklearn_predictions.append(sklearnKNN.predict([x_predict]))

        my_fold_errors.append(mean_squared_error(y_test.values.tolist(), my_predictions))
        sklearn_fold_errors.append(mean_squared_error(y_test.values.tolist(), sklearn_predictions))

    print("Errors for {} neighbours".format(k))
    print("MyKNN: {}".format(numpy.average(my_fold_errors)))
    print("SkLearn: {}".format(numpy.average(sklearn_fold_errors)))
