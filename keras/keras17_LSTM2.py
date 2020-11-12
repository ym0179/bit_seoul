#Day4
#2020-11-12

#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y = np.array([4,5,6,7])                            #(4, )
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)
x = x.reshape(x.shape[0],x.shape[1],1) #x.shape를 (4,3,1)로 바꿈 
print("x.shape : ", x.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(30, input_shape = (3,1))) 
model.add(LSTM(30, input_length=3, input_dim=1)) #input_shape는 input_length와 input_dim으로 나눌 수 있음
model.add(Dense(20))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1,verbose=2)

#4. 평가, 예측
x_input = np.array([5,6,7]) # LSTM input 3차원, (3, ) -> (1,3,1)
x_input = x_input.reshape(1,3,1)

result = model.predict(x_input) # [8] 나와야함
print(result)