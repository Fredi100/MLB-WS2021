import pandas as pd
from scipy import stats

# Data from https://www.kaggle.com/gizemaydn/iris-dataset-with-outliers/version/1
data_headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv("iris_with_outliers.csv", sep=',', names=data_headers)

print("Number of NaN entries before:")
print(data.isna().sum())

# Cleaning all NaN values by replacing them with the mean
for column in data.columns[:-1]:
    data[column].fillna(value=data[column].mean(), inplace=True)
print("\nNumber of NaN entries after:")
print(data.isna().sum())

print("\n=================================================")
print("Data before normalization:")
print(data)

# Data without target
X = data.iloc[:, :-1]
for column_name, column_data in X.iteritems():
    # Mean and Standard Deviation
    mean, std = stats.norm.fit(column_data)
    # Normalizing
    data[column_name] = column_data.apply(lambda x: (x - mean) / std)

print("\nData after normalization:")
print(data)

