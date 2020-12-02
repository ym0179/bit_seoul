#Day18
#2020-11-18

import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_dim=1, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["acc"])
model.fit(x, y, epochs=100, batch_size=1) 

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss: ", loss)
print("acc: ", acc)

y_pred = model.predict(x)
print("결과물 : \n", y_pred)
#선형회귀에서 소수값으로 예측값이 나오는데 기계는 0.999 != 1로 인식함으로 accuracy가 떨어짐
#linear regressor에서는 accuracy라는 평가지표를 쓸 수 없다
#