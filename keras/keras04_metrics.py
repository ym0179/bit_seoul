#Day2
#2020-11-10

import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss="mse", optimizer="adam", metrics=["acc"]) #loss는 기본, metrics는 추가 평가지표
model.compile(loss="mse", optimizer="adam") 
model.fit(x, y, epochs=100) 

#4. 평가, 예측
# loss, acc = model.evaluate(x, y) #2개 이상의 반환값은 리스트로 반환
loss = model.evaluate(x, y)

print("loss: ", loss)
# print("acc: ", acc)

# y_pred = model.predict(x)
# print("결과물 : \n", y_pred)


# 단축키
# shift+delete 라인 삭제
# ctrl+/ 라인 주석
# ctrl+c 라인 복사