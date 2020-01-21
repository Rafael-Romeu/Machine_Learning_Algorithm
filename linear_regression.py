# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

x, y = datasets.load_diabetes(return_X_y = True)

x = x[:, np.newaxis, 2]

trainX = x[:-20]
testX  = x[-20:]

trainY = y[:-20]
testY  = y[-20:]

regression = linear_model.LinearRegression()

regression.fit(trainX, trainY)

diabetes_y_pred = regression.predict(testX)

score = regression.score(testX, testY)

print("Data X: ", x)
print("Data Y: ", y)


print("Score: \n", score)

print("Coefficient: \n", regression.coef_)
print("Intercept: \n", regression.intercept_)

print("Mean Squared Error %.2f" % mean_squared_error(testY, diabetes_y_pred))

print("R2 Score %.2f" % r2_score(testY, diabetes_y_pred))