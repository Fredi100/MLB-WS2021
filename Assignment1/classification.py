
import pandas as pd
from KNN import KNN
from KFold import KFold
from sklearn.neighbors import KNeighborsClassifier

# Load Data
data_headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv("../iris.csv", sep=';', names=data_headers)

k = 3
num_folds = 5
mean_accuracies_myKNN = []
mean_accuracies_sklearn = []

# hyperparam tuning with odd values of k
for k in range(1, 10, 2):
    kFold = KFold(num_folds, data)

    my_accuracies = []
    sklearn_accuracies = []

    for i in range(0, num_folds):
        # get split
        x_train, x_test, y_train, y_test = kFold.get_folds(i)

        # fit classifiers
        myKNN = KNN(k, x_train, y_train)
        sklearnKNN = KNeighborsClassifier(k)
        sklearnKNN.fit(x_train, y_train)

        total = x_test.shape[0]
        correct_myKNN = 0
        correct_sklearn = 0
        for x_predict, y in zip(x_test.values.tolist(), y_test.values.tolist()):
            if myKNN.predict(x_predict) == y:
                correct_myKNN += 1
            if sklearnKNN.predict([x_predict]) == y:
                correct_sklearn += 1

        my_accuracies.append(correct_myKNN / total)
        sklearn_accuracies.append(correct_sklearn / total)

    my_average = sum(my_accuracies) / len(my_accuracies)
    sklearn_average = sum(sklearn_accuracies) / len(sklearn_accuracies)

    mean_accuracies_myKNN.append(round(my_average, 3))
    mean_accuracies_sklearn.append(round(sklearn_average, 3))

print("Acc for k = 1, 3, 5, 7, 9")
print("My KNN: {}".format(mean_accuracies_myKNN))
print("SKLearn: {}".format(mean_accuracies_sklearn))