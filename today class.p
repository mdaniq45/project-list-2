import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\hi\Downloads\logit classification.csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_test,y_train)
y_pred=classifier.predict(x_test)

from sklearn.matrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(ac)
basic=classifier.score(x_train,y_test)
print(bc)
variance
