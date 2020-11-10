#Day2
#2020-11-10

#실습
#나쁜 모델 만들기
#R2를 음수가 아닌 0.5이하로 줄이기
#레이어는 인풋과 아웃풋을 포함 7개 이상(히든이 5개 이상)
#히든레이어 노드는 레이어당 각각 최소 10개 이상
#batch_size = 1
#epochs = 100이상
#데이터 조작 금지

import numpy as np 

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(3000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(3000))
model.add(Dense(3000))
model.add(Dense(1000))
model.add(Dense(5000))
model.add(Dense(5000))
model.add(Dense(5000))
model.add(Dense(5000))
model.add(Dense(5000))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", 
            metrics=["mae"])
model.fit(x_train, y_train, epochs=100, batch_size=1) 

#4. 평가, 예측
# loss, mae = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test, batch_size=1)

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