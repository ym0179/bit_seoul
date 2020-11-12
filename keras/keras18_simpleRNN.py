#Day4
#2020-11-12


#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) 
y = np.array([4,5,6,7])                            

x = x.reshape(x.shape[0],x.shape[1],1)
# x = x.reshape(4,3,1)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
#simpleRNN gate가 없음
model = Sequential()
model.add(SimpleRNN(10, activation = 'relu', input_shape = (3,1)))
model.add(Dense(20))
model.add(Dense(1))

model.summary()
'''
계산방법1. 
g: no. of FFNNs in a unit (RNN has 1, GRU has 3, LSTM has 4) *****
h: size of hidden units
i: dimension/size of input

Every FFNN has h(h+i) + h parameters
num_params = g × [h(h+i) + h]

model.add(LSTM(30, input_shape = (3,1))) => 1*(30(30+1)+30)
model.add(LSTM(10, input_shape = (3,1))) => 1*(10(10+1)+10)


============================================================
계산방법2.

model.add(LSTM(30, input_shape = (3,1))) => 1*(1+1+10)*10
(몇개씩 잘라서 작업 + bias(=1) + 다음 레이어 노드 수) * 다음 레이어 노드 수
'''

'''
#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1,verbose=2)


#4. 평가, 예측
x_input = np.array([5,6,7]) # LSTM input 3차원
# (3, ) -> (1,3,1)
x_input = x_input.reshape(1,3,1)

result = model.predict(x_input) # [8] 나와야함
print(result)
'''