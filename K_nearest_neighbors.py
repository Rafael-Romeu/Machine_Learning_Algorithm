# K-NN
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

irisDataset = datasets.load_iris()

irisFeatures = irisDataset.data
irisTarget   = irisDataset.target

xTrain, xTest, yTrain, yTest = train_test_split(irisFeatures, irisTarget, test_size=0.2)

from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=3, algorithm='ballTree')
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)

from sklearn.metrics import confusion_matrix, f1_score

confusion_matrix = confusion_matrix(yTest, yPred)
f1_score         = f1_score(yTest, yPred, average='weighted')

print("Confusion Matrix: \n", confusion_matrix)
print("F1-Score: ", f1_score)