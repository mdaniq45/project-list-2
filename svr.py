import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


dataset = pd.read_csv(r"C:\Users\hi\Downloads\2 september\30th\1.POLYNOMIAL REGRESSION\emp_sal.csv")


X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values
from sklearn.svm import SVR
svr_regressor=SVR()
svr_regressor.fit(x,y)

