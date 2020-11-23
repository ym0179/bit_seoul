#Day11
#2020-11-23
#회귀문제

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x,y = load_diabetes(return_X_y=True)
# print(y)
dataset = load_diabetes()
print(dataset['feature_names'])
'''
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
'''

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# scale
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
# model = LinearSVC() #회귀문제 성능 엄청 떨어짐
# model = SVC() #회귀문제 성능 엄청 떨어짐
# model = KNeighborsClassifier() #회귀문제 성능 엄청 떨어짐
model = KNeighborsRegressor()
# model = RandomForestClassifier() #회귀문제 성능 엄청 떨어짐
# model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 예측, 평가
score = model.score(x_test, y_test)
print("model.score : ", score)
# accuracy_score를 넣어서 비교할 것
# 회귀 모델인 경우 r2_score 와 비교할 것

y_predict = model.predict(x_test)
# acc = accuracy_score(y_test,y_predict)
# print("acc : ", acc)
r2 = r2_score(y_test,y_predict)
print("r2 score : ", r2)

print(y_test[:10], "의 예측 결과\n",y_predict[:10])

''' 모델별 결과
LinearSVC() #회귀문제 성능 엄청 떨어짐
model.score :  0.011235955056179775
r2 score :  -0.15630588545158997

SVC() #회귀문제 성능 엄청 떨어짐
model.score :  0.0
r2 score :  -0.0833765078750015

KNeighborsClassifier() #회귀문제 성능 엄청 떨어짐
model.score :  0.0
r2 score :  -0.5564141962639386

RandomForestClassifier() #회귀문제 성능 엄청 떨어짐
model.score :  0.011235955056179775
r2 score :  -0.07306857875711414

KNeighborsRegressor()
model.score :  0.38626977834604637
r2 score :  0.38626977834604637

RandomForestRegressor()
model.score :  0.4093733818689488
r2 score :  0.4093733818689488
'''
