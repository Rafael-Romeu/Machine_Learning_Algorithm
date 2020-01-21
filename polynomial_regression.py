# Solve SSL problems
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

x = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.DataFrame(data.target, columns = ["MedInc"])

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

poly = PolynomialFeatures(degree = 3) 

xPoly = poly.fit_transform(x)

pol_reg = LinearRegression()
pol_reg.fit(xPoly, y)

predictions = pol_reg.predict(xPoly)
score = pol_reg.score(xPoly, y)

print("Score: ", score)
print("Coefficient: \n", pol_reg.coef_)
print("Mean Squared Error %.2f" % mean_squared_error(y, predictions))
print("R2 Score %.2f" % r2_score(y, predictions))
