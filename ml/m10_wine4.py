#Day11
#2020-11-23

#winequality-white.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# 1. 데이터
#pandas로 csv 불러오기
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')

#x,y 값 나누기
y = wine['quality']
x = wine.drop('quality',axis=1)
# print(x)
# print(y)
# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

#인위적으로 y의 labeling 값을 조절 (분포를 더 작게 잡아주기)
#quality 0-4까지는 0, 5-7은 1, 8-9는 2
newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

# print(newlist)
y = newlist

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

# scale
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor() #분류 문제 에러
model = RandomForestClassifier()
# model = RandomForestRegressor() #분류 문제 에러

# 3. 훈련
model.fit(x_train,y_train)

# 4. 예측, 평가
score = model.score(x_test, y_test)
print("model.score : ", score)
# accuracy_score를 넣어서 비교할 것
# 회귀 모델인 경우 r2_score 와 비교할 것

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("acc : ", acc)

print(y_test[:10], "의 예측 결과\n",y_predict[:10]) #1이 가장 많음

'''
model.score :  0.9469387755102041
acc :  0.9469387755102041
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 의 예측 결과 #대부분 1이라 1예측.. 데이터 조작이 될수도;;
 [1 1 1 1 1 1 1 1 1 1]
'''