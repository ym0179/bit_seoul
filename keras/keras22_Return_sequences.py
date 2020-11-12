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


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(35,activation='relu',input_shape=(3,1), return_sequences=True)) 
#(None, 35)를 출력 
#입력한 차원을 그대로 그 다음으로 전달
model.add(LSTM(25,activation='relu')) #여기에 return_sequences 해주면 dense에 3차원 넣어줌
#LSTM을 여러개 하는게 나을까?
#데이터의 구조, 모델 구성에 따라 다른데 좋을 수도 있고 나쁠 수도 있음
#첫번째 LSTM에서 시계열 데이터가 넘어온다면 (순차적 데이터) 더 좋을 수도,,
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
x_input = x_input.reshape(1,3,1)
result = model.predict(x_input)
# print("x",x_input)
print("result : ",result)

loss = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)