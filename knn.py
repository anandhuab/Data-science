# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:52:49 2022

@author: DELL
"""

from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
print(iris.feature_names)
print("class names: \n",iris.target_names)
print("0=setosa,1=versicolor,2=virginica")
print(iris.target)
print(iris.data.shape)
x,y=iris.data[:,:],iris.target
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
print("x_train:",x_train.shape)
print("y_train:",y_train.shape)
print("y_train:",x_test.shape)
print("y_test:",y_test.shape)
print(x_train)
print("\n")
print(x_test)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print("prediction of species: {}".format(y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix,classification_report
from matplotlib import numpy
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))