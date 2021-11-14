########################################################################################################################
# EX1
########################################################################################################################


'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutff value for idnetifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers!
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''

########################################################################################################################
# Solution
########################################################################################################################
import numpy as np
import pandas as pd
from util import delimiter_fix, plot_outliers
from scipy.stats import norm
from sklearn.impute import KNNImputer
from sklearn.datasets import load_wine

if __name__ == '__main__':
    # set pandas options to make sure you see all info when printing dfs
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    missing_types = [np.NAN, 'missing', -999]

    # load file and correct delimiters ... delimiter_fix() fixes all faulty lines but one (off by 1 entry)
    file = delimiter_fix('wine_exercise.csv')

    # load data (drop bad lines due to imperfect delimiter fix)
    data = pd.read_csv(file, delimiter=';', error_bad_lines=False, skiprows=1, skipfooter=1)

    # consistently encode NA values (to numpy NaN)
    data = data.replace(missing_types, np.NAN)

    # fix and encode values in col 'season' (SPRING => 0, etc.)
    data = data.replace(
        to_replace=[r'(?i)(spr)', r'(?i)(sum)', r'(?i)(aut)', r'(?i)(win)'],
        value=[0, 1, 2, 3],
        regex=True)

    # split country-age into respective cols
    data = data.rename(columns={'country-age': 'country'})
    data['age'] = 0
    for i, val in enumerate(data.iloc[:, -2].values.tolist()):
        split = str(val).split('-')
        data.iloc[i, -2] = split[0]         # country
        data.iloc[i, -1] = split[1][0]      # age (remove 'years')

    # remove duplicate rows
    data = data.drop_duplicates()

    # remove outliers
    # proline
    data.loc[:, 'proline'] = data.loc[:, 'proline'].replace(to_replace=0, value=np.NAN)

    # fix magnesium and total_phenols in row 165 by hand to be able to continue
    data.iloc[165, 4] = 111
    data.iloc[165, 5] = 1.7

    # magnesium
    magnesium_vals = [float(string) for string in data.loc[:, 'magnesium']]
    # plot
    #plot_outliers(magnesium_vals)
    # remove outliers at cutoff point 0.025%
    mu, std = norm.fit(magnesium_vals)
    ppf = norm.ppf(1 - 0.00025, mu, std)
    for i, val in enumerate(data.loc[:, 'magnesium']):
        if float(val) > ppf:
            data.loc[i, 'magnesium'] = np.NAN

    # total_phenols
    phenols_vals = [float(string) for string in data.loc[:, 'total_phenols']]
    # plot
    #plot_outliers(phenols_vals)
    # remove outliers at cutoff point 0.025%
    mu, std = norm.fit(phenols_vals)
    ppf = norm.ppf(1 - 0.00025, mu, std)
    for i, val in enumerate(data.loc[:, 'total_phenols']):
        if float(val) > ppf:
            data.loc[i, 'total_phenols'] = np.NAN

    # impute missing values
    col_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    # create dataframe from sklearn wine dataset
    data_sklearn = pd.DataFrame(load_wine()['data'], columns=col_names)
    # fit imputer on sklearn data, then transform data
    imputer = KNNImputer(n_neighbors=2)
    imputer.fit(data_sklearn)
    data.iloc[:, :-5] = imputer.transform(data.iloc[:, :-5])


    print(data)
    exit()