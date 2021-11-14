import re
from io import StringIO
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

missing_strings = ['nan', 'missing', '-999']
# find and replace ',' intended as '.' per col
regexStrs = [r'\d+[.]\d+', r'\d+[.]\d+', r'\d+[.]\d+', r'\d+[.]\d+', r'\w+', r'\d+[.]\d+', r'\d+[.]\d+',
             r'\d+[.]\d+', r'\d+[.]\d+', r'\d+[.]\d+', r'\d+[.]\d+', r'\d+[.]\d+', r'\w+', r'\w+', r'\w+', r'\w+',
             r'\w+[\-]?\w+']


def delimiter_fix(path):
    replaced = ''
    with open(path, newline='') as file:
        broken_index = -1
        fixed_lines = []
        for i, line in enumerate(file.readlines()):
            # split at ,
            split = line.split(',')
            # only look at lines that had ','
            if len(split) > 1:
                broken_index += 1
                fixed_lines.append('')
                regex_index = 0
                for j, e in enumerate(split):
                    # filter missing data
                    if e in missing_strings:
                        regex_index += 1
                        fixed_lines[broken_index] += e + ';'
                    # filter correct formats
                    elif re.search(regexStrs[regex_index], e):
                        regex_index += 1
                        fixed_lines[broken_index] += e + ';'
                    else:
                        e_split = e.split(';')
                        # filter cases such as '83;1'
                        if len(e_split) > 1:
                            regex_index += 1
                            if j + 1 == len(split):
                                break
                            post_decimal = split[j + 1].split(';')[0]
                            fixed_lines[broken_index] += e_split[1] + '.' + post_decimal + ';'
                        else:
                            if j + 1 == len(split):
                                break
                            regex_index += 1
                            post_decimal = split[j + 1].split(';')[0]
                            fixed_lines[broken_index] += e + '.' + post_decimal + ';'

                replace_end = len(fixed_lines[broken_index])
                line = fixed_lines[broken_index][:-1] + line[replace_end:]
            replaced += line
    return StringIO(replaced)

def plot_outliers(values):
    mu, std = norm.fit(values)
    # Plot the histogram.
    plt.hist(values, bins=25, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()