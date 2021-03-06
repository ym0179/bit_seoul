#Day1 
#2020-11-09

import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5]) #정제된 데이터

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential() #순차적으로 위에서 부터 밑에까지 연산
#연산 하나하나가 y = wx + b, 각각 최적의 w값 구함
#곱셈의 확률
model.add(Dense(300, input_dim=1)) #input dimension: 1개가 입력이 됨
#Dense 단순 DNN
model.add(Dense(5000))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))
#하이퍼파라미터 튜닝

#3. 컴파일, 훈련 (컴퓨터가 모델을 알아듣게)
model.compile(loss="mse", optimizer="adam", metrics=["acc"])
#loss가 낮을수록 좋음
#mean squared error 평균제곱오차 최적화 
#최적화 함수 adam
#평가방식 accuracy

# model.fit(x, y, epochs=100, batch_size=1) #정제된 데이터 모델에게 줌
#epochs 100번 훈련
#batch size 1 - 한개씩 잘라서 넣는다
#batch size default = 32
#batch size가 전체 데이터 수보다 큰 경우에는 자동으로 전체 데이터 수로 잡음


#4. 평가, 예측
# loss, acc = model.evaluate(x, y, batch_size=1)

#without batch_size
model.fit(x, y, epochs=1000)
loss, acc = model.evaluate(x, y)

print("loss: ", loss)
print("acc: ", acc)