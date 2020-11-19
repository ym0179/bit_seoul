#Day9
#2020-11-19

from sklearn.datasets import load_iris
import numpy as np

iris=load_iris()
print(iris)
print(type(iris))

x_data=iris.data
y_data=iris.target

print(type(x_data))
print(type(y_data))

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)