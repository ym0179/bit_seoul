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
scores = cross_val_score(model, x_train, y_train, cv=kfold) #cv에 바로 '5'이렇게 써도됨
print('SVC : ',scores, "\n평균 : ", scores.mean())
# SVC :  [0.95833333 1.         1.         0.91666667 1.        ]
# 평균 :  0.975

model = LinearSVC()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('LinearSVC : ',scores, "\n평균 : ", scores.mean())
# LinearSVC :  [0.79166667 0.91666667 0.875      0.91666667 0.83333333]
# 평균 :  0.8666666666666666

model = KNeighborsClassifier()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('KNeighborsClassifier : ',scores, "\n평균 : ", scores.mean())
# KNeighborsClassifier :  [1.         0.91666667 0.95833333 1.         1.        ]
# 평균 :  0.975

model = KNeighborsRegressor() #분류 문제임으로 추천 X -> cross_val_score가 회귀 모델은 자동으로 r2 계산함으로 에러는 안남
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('KNeighborsRegressor : ',scores, "\n평균 : ", scores.mean())
# KNeighborsRegressor :  [0.98930362 0.95801193 0.99710843 0.90873239 0.98878505]
# 평균 :  0.9683882848858898

model = RandomForestClassifier()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('RandomForestClassifier : ',scores, "\n평균 : ", scores.mean())
# RandomForestClassifier :  [1. 1. 1. 1. 1.]
# 평균 :  1.0

model = RandomForestRegressor() #분류 문제임으로 추천 X -> cross_val_score가 회귀 모델은 자동으로 r2 계산함으로 에러는 안남
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('RandomForestRegressor : ',scores, "\n평균 : ", scores.mean())
# RandomForestRegressor :  [0.987368   0.99996739 0.999075   0.99999368 0.99221446]
# 평균 :  0.9957237066692398