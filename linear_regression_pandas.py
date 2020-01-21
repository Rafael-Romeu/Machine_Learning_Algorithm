# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns = data.feature_names)

target = pd.DataFrame(data.target, columns = ["MEDV"])

lm = linear_model.LinearRegression()
model = lm.fit(df, target)

predictions = lm.predict(df)

score = lm.score(df, target)

print("Score: \n", score)

print("Coefficient: \n", lm.coef_)

print("Mean Squared Error %.2f" % mean_squared_error(target, predictions))

print("R2 Score %.2f" % r2_score(target, predictions))