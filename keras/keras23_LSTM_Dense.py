#Day4
#2020-11-12

#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
print(y.shape)
x_input = np.array([50,60,70]) #(3,)
x_input = x_input.reshape(1,3) #x와 shape 동일하게 맞춰주기


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
# model.add(Dense(35,activation='relu',input_dim=3)) 
model.add(Dense(35,activation='relu',input_shape=(3,))) 
model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss="mse",optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=15,mode='auto')
model.fit(x,y,epochs=1000,batch_size=1,verbose=2,callbacks=[es]) 

#4. 예측
result = model.predict(x_input)
print("result : ",result)

loss = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)