#Day2
#2020-11-10

import numpy as np 

#1. 데이터
#학습시킨 데이터로 평가하면 안됨
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) #훈련할 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) #평가할 데이터
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) #예측할 데이터

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
model.fit(x_train, y_train, epochs=100) 

#4. 평가, 예측
# loss, mae = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test)

print("loss: ", loss)

y_pred = model.predict(x_pred)
print("결과물 : \n", y_pred)
