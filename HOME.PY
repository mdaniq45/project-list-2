import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import sklearn.model_selection import 

dataset=pd.read_csv(r"C:\Users\hi\Downloads\Investment.csv")
dataset

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#== we build mlr model

m = regressor.coef_
m

c = regressor.intercept_
c
X=np.append(arr=np.one((50,1)).astype(int),values=x)
import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
