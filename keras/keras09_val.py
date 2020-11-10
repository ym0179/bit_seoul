#Day2
#2020-11-10

# validation 안했을 때
# RMSE :  0.5123874269447404
# R2 :  0.8687295623544742

# validation 했을 때
# RMSE :  0.2575600393898422
# R2 :  0.9668314130547515

import numpy as np 

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_val = np.array([11,12,13,14,15]) #검증 데이터 validation
y_val = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18])
x_test = np.array([16,17,18,19,20])
y_test = np.array([16,17,18,19,20])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", 
            metrics=["mae"])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100) 

#4. 평가, 예측
# loss, mae = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)

# y_predict = model.predict(x_pred)
y_predict = model.predict(x_test)

print("결과물 : \n", y_predict)

# 실습 : 결과물 오차 수정. 미세조정
from sklearn.metrics import mean_squared_error
#사용자 정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2) # max 값: 1