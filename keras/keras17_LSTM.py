#Day4
#2020-11-12

#RNN 순환신경망 - 순차적 모델
#LSTM

#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3) 행렬
y = np.array([4,5,6,7])                            #(4, ) 벡터
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)
x = x.reshape(x.shape[0],x.shape[1],1) #x.shape를 (4,3,1)로 바꿈 
#(4,3,1)과 (4,3)의 전체 데이터 수가 동일함으로 reshape 가능
# x = x.reshape(4,3,1)
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(30, activation = 'relu', input_shape = (3,1)))
#(행,열,몇개씩 자르는지) -> 3차원 데이터로 바꿔줘야함
#(행,열,몇개씩 자르는지) -> 마지막에 LSTM 만들 때 한개씩 잘라서 연산하겠다는게 명시됨 
#like [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]] = (4,3,1)
model.add(Dense(20))
model.add(Dense(1))

model.summary()
'''
Feed-Forward Neural Network (FNN) 순방향 신경망
FNN 구조: input -> hidden layer 1 -> hidden layer 2 -> ... -> hidden layer k -> output

LSTM 구조: 오랜 기간동안 정보를 기억하는 특징, 체인구조
=> 4개의 상호작용하는 레이어가 있는 반복되는 모듈
- The input gate
- The forget gate
- Updating the cell state
- The output gate

계산방법1. 
g: no. of FFNNs in a unit (RNN has 1, GRU has 3, LSTM has 4) *****
h: size of hidden units
i: dimension/size of input

Every FFNN has h(h+i) + h parameters
num_params = g × [h(h+i) + h]

model.add(LSTM(30, input_shape = (3,1))) => 4*(30(30+1)+30)
model.add(LSTM(10, input_shape = (3,1))) => 4*(10(10+1)+10)

https://brunch.co.kr/@chris-song/9
============================================================
계산방법2.

model.add(LSTM(30, input_shape = (3,1))) => 4*(1+1+10)*10
4(gate 총 4개) * (몇개씩 잘라서 작업 + bias(=1) + 다음 레이어 노드 수) * 다음 레이어 노드 수

* LSTM은 연산양이 많음 => 속도 느려짐
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