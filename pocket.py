import math
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt


def error(w, data, targets):
    cols = list(targets.columns.values)
    error = 0

    for index in range(len(targets)):
        if (np.dot(w, data[index, :]) >= 0 and targets[cols[0]].iloc[index] == 'M') or (np.dot(w, data[index, :]) < 0 and targets[cols[0]].iloc[index] == 'B'):
            error += 1
    return error

def pocketstep(w, index, data, targets, olderror):
    cols = list(y.columns.values)

    if (np.dot(w, data[index, :]) < 0 and targets[cols[0]].iloc[index] == 'M') or (np.dot(w, data[index, :]) >= 0 and targets[cols[0]].iloc[index] == 'B'):
        return w, olderror, False
    neww = w
    if targets[cols[0]].iloc[index] == 'M':
        neww -= data[index, :]
    else:
        neww += data[index, :]
    newerror = 0#error(neww, data, targets)
    if newerror < olderror:
        return neww, newerror, True
    return w, olderror, False


# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


cols = list(X.columns.values)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

indeces = [20, 27, 21, 15, 7, 23, 26, 13, 24, 10, 4, 25, 2, 14, 5, 3, 6, 9, 1, 8]

fig = clf.fit(X,y)

sizes = [5, 10, 15, 20]

print(y_train)

for size in sizes:
    names = []
    for j in range(size):
        names.append(cols[indeces[j]])
    currX_train = X_train[names]
    currX_test = X_train[names]

    currX_train['bias'] = 1
    currX_test['bias'] = 1

    maximum = len(currX_train)

    currX_train = currX_train.to_numpy().astype(float)

    count = 0
    index = 0
    olderror = 345432222453
    w = np.zeros(size + 1, dtype = float)
    print(y_train)
    while count < maximum:
        w, olderror, changed = pocketstep(w, index, currX_train, y_train, olderror)
        count += 1
        if changed:
            count = 0
        index = (index + 1) % maximum
    print("{}/{}".format(error(w, currX_test.to_numpy(), y_test), len(y_test)))
