
#import laberary
import numpy as np

import pandas as pd
import sklearn
#import datasets
df=pd.read_csv(r"C:\Users\hi\OneDrive\Desktop\csv files\sample_submission.csv")
#chakes the data frame
df
df.head()
#chake the information
df.info()
#chakes the null values
df.isnull().sum()
#drop any null valuse
df.dropna(inplace=True)
#predict  and target
#create matrice of pridiction
x=df.iloc[:,1:8]
#creating target
y=df.iloc[:,0]
x=pd.get_dummies(x)
x
#use train test split methods
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#initilize and train and the linear regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
#lr_model=linearRegression()
#lr_model.fit(x_train,y_train)
#make predict on the data sets
y_pred=lr_model.predict(x_test)
# evaluate the model
#mse=men_squared_error(y_test,y_pred)
#print(f"mean square error:{mse}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

