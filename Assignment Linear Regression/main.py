from sklearn.datasets import make_regression
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression

if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=1, noise=25)

    # fit and get params
    reg = LinearRegression().fit(X, y)
    k = reg.coef_[0]
    d = reg.intercept_

    # plot samples and regression line
    x = np.linspace(-5, 5, 100)
    f_x = k * x + d
    plt.plot(x, f_x, '-r')
    plt.scatter(X, y)
    plt.show()

    # plot histogram
    plt.hist(X, bins=10)
    plt.show()

    # calculate MSE, R2 and p-value
    n = len(X)
    predictions = reg.predict(X)
    # residual sum of squares
    rss = np.sum((predictions - y) ** 2)
    tss = np.sum((np.mean(y) - y) ** 2)
    mse = rss / n
    r_2 = 1 - (rss / tss)
    # p-value
    _, p_value = f_regression(X, y)

    print('MSE:', mse)
    print('R2:', r_2)
    print('p-value:', p_value[0])