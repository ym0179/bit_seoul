#Day11
#2020-11-23
#회귀문제

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x,y = load_boston(return_X_y=True)
# print(y)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# scale
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
# model = LinearSVC() #회귀문제 에러
# model = SVC() #회귀문제 에러
# model = KNeighborsClassifier() #회귀문제 에러
model = KNeighborsRegressor()
# model = RandomForestClassifier() #회귀문제 에러
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
LinearSVC()  #회귀문제 에러
SVC()  #회귀문제 에러
KNeighborsClassifier() #회귀문제 에러
RandomForestClassifier() #회귀문제 에러

KNeighborsRegressor()
model.score :  0.8404010032786686
r2 score :  0.8404010032786686

RandomForestRegressor()
model.score :  0.9232854563760624
r2 score :  0.9232854563760624
'''
