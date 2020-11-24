#Day12
#2020-11-24

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
# print(iris)
x = iris.iloc[:,:4]
y = iris.iloc[:,-1]

# print(x.shape, y.shape) #(150, 4) (150,)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

model = SVC()
scores = cross_val_score(model, x_train, y_train, cv=kfold) #분류일 때는 accuracy, 회귀일 때는 r2 score로 나옴
print('SVC : ',scores)
# SVC :  [1.         1.         0.875      0.95833333 1.        ]

model = LinearSVC()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('LinearSVC : ',scores)
# LinearSVC :  [0.75       0.66666667 0.75       0.66666667 0.91666667]

model = KNeighborsClassifier()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('KNeighborsClassifier : ',scores)
# KNeighborsClassifier :  [1.         0.95833333 1.         1.         1.        ]

model = KNeighborsRegressor() #분류 문제임으로 추천 X
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('KNeighborsRegressor : ',scores)
# KNeighborsRegressor :  [1.         0.99747368 0.97726316 0.9808547  0.97391304]

model = RandomForestClassifier()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('RandomForestClassifier : ',scores)
# RandomForestClassifier :  [1.         1.         1.         1.         0.95833333]

model = RandomForestRegressor() #분류 문제임으로 추천 X
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('RandomForestRegressor : ',scores)
# RandomForestRegressor :  [0.99969577 0.9999925  0.93750769 0.99131588 0.99309484]
