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
model.add(Dense(300, input_dim=1))
model.add(Dense(5000))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
#lr Default 0.001
# optimizer = Adam(lr=0.001) #loss :  3.154809855195062e-13 결과물 :  [[11.000001]]
# optimizer = Adadelta(lr=0.001) #loss :  0.0002002382680075243 결과물 :  [[11.003489]]
# optimizer = Adamax(lr=0.001) #loss :  7.807443762430921e-05 결과물 :  [[11.011876]]
# optimizer = Adagrad(lr=0.001) #loss :  5.532633622351568e-06 결과물 :  [[10.996891]]
# optimizer = RMSprop(lr=0.001) #loss :  0.13966414332389832 결과물 :  [[11.401719]]
# optimizer = SGD(lr=0.001) #loss :  2.872389814001508e-06 결과물 :  [[10.996484]]
optimizer = Nadam(lr=0.001) #loss :  0.00013261320418678224 결과물 :  [[11.01114]]


model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
model.fit(x, y, epochs=100, batch_size=1) 

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss: ", loss)

y_pred = model.predict([11])
print("loss : ",loss, "결과물 : ", y_pred)