#Day4
#2020-11-12

#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input = np.array([50,60,70])
print(x.shape)
x = x.reshape(13,3,1)

# 실습 LSTM 완성하시오
# 예측값 80

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(35,activation='relu',input_shape=(3,1)))
model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

# model.add(LSTM(30,activation='relu',input_shape=(3,1)))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

model.compile(loss="mse",optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1,verbose=2) #LSTM은 파라미터가 많기 때문에 충분한 훈련량 필요

#4. 예측
x_input = x_input.reshape(1,3,1)
result = model.predict(x_input)
# print("x",x_input)
print("result : ",result)
