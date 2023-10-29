import pandas as pd
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression

salary = pd.read_csv("salary.csv")
salary.columns
X = salary[['Experience_Years']]
y = salary['Salary']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=250)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=None)

y_pred = model.predict(X_test)
print(y_pred)


mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
mean_squared_error(y_test,y_pred)