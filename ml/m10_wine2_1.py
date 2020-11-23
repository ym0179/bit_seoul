#Day11
#2020-11-23

#winequality-white.csv
#ML 모델별 만들기

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
#pandas로 csv 불러오기
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
# print(datasets)
# print(datasets.shape) #(4898, 12)

# pandas dataframe를 numpy 배열로 변환하기
datasets = datasets.to_numpy()

# x,y 데이터 나누기
x = datasets[:,:-1]
y = datasets[:,-1]
# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

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

print(y_test[:10], "의 예측 결과\n",y_predict[:10])
