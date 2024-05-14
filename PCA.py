from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np

#previously normalized MNIST dataset (see kmeans)
X = pd.read_csv('normalized.csv')

X = X.to_numpy().transpose()

sum = X[:,0]

print(X[:,0].shape)

for i in range(1, 27914):
    sum += X[:,i]/27914

sum = sum.reshape(784,1)

print(sum.T.shape)

S = np.matmul(((X[:,0]/27914) - sum).T, ((X[:,0]/27914) - sum))


for i in range(1, 27914):
    S += np.matmul((X[:,i] - sum), (X[:,i] - sum).transpose()) / 27914
    print("{}/{}".format(i + 1,27914))

S = pd.read_csv('S.csv')
print(S)
S = S.to_numpy()
S = S[:,1:]
eigenvalues, eigenvectors = np.linalg.eig(S)

print(eigenvalues[500])
df = pd.DataFrame(eigenvectors)
df.to_csv("./eigs.csv")


