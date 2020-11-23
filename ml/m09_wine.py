#Day11
#2020-11-23
#다중분류문제

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x,y = load_wine(return_X_y=True)
dataset = load_wine()
# print(y)
print(dataset['feature_names'])
print(dataset.feature_names)
print(dataset.target_names) #['class_0' 'class_1' 'class_2']
'''
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 
'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
'''

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

''' 모델별 결과
LinearSVC()
model.score :  0.9722222222222222
acc :  0.9722222222222222

SVC()
model.score :  1.0
acc :  1.0

KNeighborsClassifier()
model.score :  1.0
acc :  1.0

RandomForestClassifier()
model.score :  1.0
acc :  1.0

KNeighborsRegressor()  #분류 문제 에러
RandomForestRegressor()  #분류 문제 에러
'''