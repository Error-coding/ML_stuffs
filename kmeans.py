from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import re


def checkEuclidian(centroids, vector):
    dist = 348095789347895.0
    index = -1

    for i in range(len(centroids)):
        dista = np.linalg.norm(centroids[i]-vector)
        if dista < dist:
            index = i
            dist = dista
    return index

def checkCosineSim(centroids, vector):
    dist = -2
    index = -1

    for i in range(len(centroids)):
        dista = np.dot(vector, centroids[i])/(np.linalg.norm(vector)*np.linalg.norm(centroids[i]))
        if dista > dist:
            index = i
            dist = dista
    return index

# fetch data and transform to numpy array
X,y = fetch_openml('mnist_784', version=1, as_frame=True, parser='pandas', return_X_y=True)

used_classes = ['2','3','8','9']

#X = (X[y.isin(used_classes)])
y = (y[y.isin(used_classes)])
y = y.to_numpy()

cols = list(X.columns.values)

stats=pd.DataFrame()
stats["mean"] = X.mean()
stats["variance"] = X.var()

print(stats)

for i in range(len(cols)):
    print(i)
    if stats["variance"][cols[i]] < 0.00001:
        continue
    X[cols[i]] = X[cols[i]].apply(lambda x: (x - stats["mean"][cols[i]]) / math.sqrt(stats["variance"][cols[i]]))

print("Loaded normalized dataset")



Xn = X.to_numpy().transpose()

print(Xn.shape)

i1 = np.where(y == "2")[0][0]
i3 = np.where(y == "3")[0][0]
i2 = np.where(y == "8")[0][0]
i4 = np.where(y == "9")[0][0]



# select first found member of each class as centroid
centroids = [Xn[:,i1],
             Xn[:,i2],
             Xn[:,i3],
             Xn[:,i4]]

cluster = [[],[],[],[]]
oldcluster = []

for k in range(1000):
    print(k)
    cluster = [[],[],[],[]]
    
    # compute all points belonging to centroids
    cenchildren = [0,0,0,0]
    cencount = [0,0,0,0]
    amount = len(X)
    for i in range(len(X)):
        #replace for euclidian
        closest = checkCosineSim(centroids, Xn[:,i])
        cenchildren[closest] += Xn[:,i]
        cencount[closest] += 1
        cluster[closest].append(i)
    

    # update centroids
    for i in range(4):
        centroids[i] = cenchildren[i] / cencount[i]
    for a in range(len(centroids)):
        df = pd.DataFrame(centroids[a])
        df.to_csv("./cosinecen{}{}.csv".format(k + 1,a))

    breakcond = True
    if not k == 0:
        for i in range(4):
            if len(oldcluster[i]) == len(cluster[i]):
                for j in range(len(cluster[i])):
                    if not oldcluster[i][j] == cluster[i][j]:
                        breakcond = False
                        break
            else:
                print("{}:{}".format(i,len(oldcluster[i]) - len(cluster[i])))
                breakcond = False
    else:
        breakcond = False
    if breakcond:
        break
    oldcluster = cluster
        
cenchildren = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
indeces = [0, 1, 0, 1, 4, 5, 6, 7, 2, 3]
for i in range(len(X)):
    #replace for euclidian
    closest = checkCosineSim(centroids, Xn[:,i])
    cenchildren[closest][indeces[int(y[i])]] += 1

for i in range(4):
    print("Centroid {}".format(i))
    print("{}: {}".format(2,cenchildren[i][0]))
    print("{}: {}".format(3,cenchildren[i][1]))
    print("{}: {}".format(8,cenchildren[i][2]))
    print("{}: {}".format(9,cenchildren[i][3]))
    
