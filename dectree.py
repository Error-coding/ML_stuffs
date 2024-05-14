import math
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt




# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=0, max_depth = 2)
fig = clf.fit(X,y)
tree.plot_tree(fig)
plt.show()
"""for i in range(1, len(X.columns)):
    clf = DecisionTreeClassifier(random_state=0, max_depth = i)
    print(i)
    print(cross_val_score(clf, X, y, cv=5))"""
