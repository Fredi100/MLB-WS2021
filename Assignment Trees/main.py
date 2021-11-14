from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def predict(data, root, sample):
    split = root[0]
    col = split[0]
    pivot = split[1]
    confidences = root[1]
    confidences_smaller = confidences[0]
    confidences_larger = confidences[1]

    predicted_confidences = ()
    if sample[col] <= pivot:
        predicted_confidences = confidences_smaller
    else:
        predicted_confidences = confidences_larger

    prediction = predicted_confidences.index(max(predicted_confidences))
    confidence = max(predicted_confidences)

    print('the sample of class', int(sample['target']), 'belongs to class', prediction, 'with confidence', confidence)


def calc_gini(data, column, values):
    # count target occurences
    counts = [0, 0]
    for val in values:
        # value could occur multiple times, hence 'indices'
        indices = data[data[column] == val][column].index.values.tolist()
        for i in indices:
            if data.iloc[i, -1] == 0:
                counts[0] += 1
            else:
                counts[1] += 1
    total = counts[0] + counts[1]
    gini = 1 - (counts[0] / total) ** 2 - (counts[1] / total) ** 2
    confidences = (counts[0] / total, counts[1] / total)
    return gini, total, confidences


def calc_gini_total(data, column, pivot):
    values_smaller = data[data[column] <= pivot][column]
    values_larger = data[data[column] > pivot][column]

    gini_smaller, total_smaller, confidences_smaller = calc_gini(data, column, values_smaller)
    gini_larger, total_larger, confidences_larger = calc_gini(data, column, values_larger)

    total_total = total_smaller + total_larger
    gini_total = gini_smaller * (total_smaller / total_total) + gini_larger * (total_larger / total_total)

    return (column, pivot, gini_total), (confidences_smaller, confidences_larger)


if __name__ == '__main__':
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, n_samples=100)
    data = pd.DataFrame(data=np.c_[X, y], columns=['X1', 'X2', 'target'])

    plt.scatter(data['X1'], data['X2'], marker='o', c=y,
                s=25, edgecolor='k')
    plt.show()

    potential_splits = []
    for col in data.iloc[:, :-1].columns:
        # calculate means of adjacent (sorted) values
        colSorted = sorted(data[col])
        for x_cur, x_next in zip(colSorted, colSorted[1:]):
            potential_splits.append((col, (x_cur + x_next) / 2))

    ginis = []
    for split in potential_splits:
        ginis.append(calc_gini_total(data, split[0], split[1]))

    split, confidences = min(ginis, key=lambda x: x[0][2])

    print('best split at', split[1], 'of column', split[0], 'with gini index of', split[2])

    predict(data, (split, confidences), data.iloc[4])
