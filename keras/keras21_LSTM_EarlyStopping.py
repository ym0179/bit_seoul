#Day4
#2020-11-12

#너무 많은 Epoch 은 overfitting 을 일으킨다. 
#하지만 너무 적은 Epoch 은 underfitting 을 일으킨다.
#overfitting: 오차가 순조롭게 줄어들다가 어느 시점부터 오차가 커져가는 것
#Early Stopping (학습 조기 종료): 이전 epoch 때와 비교해서 오차가 증가했다면 학습을 중단하는 방법


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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
input1 = Input(shape=(3,1))
lstm1 = LSTM(35,activation='relu')(input1)
dense1 = Dense(20,activation='relu')(lstm1)
dense2 = Dense(15,activation='relu')(dense1)
dense3 = Dense(7,activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

# model = Sequential()
# model.add(LSTM(30,activation='relu',input_shape=(3,1)))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse",optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss',patience=100, mode='min')
es = EarlyStopping(monitor='loss',patience=15, mode='auto')
# patience: 오차를 보기 위해 과거 몇 epoch까지 거슬러 올라갈 것인가
# performance measure가 최소화 시켜야하는 것이면 mode를 min 으로, 최대화 시켜야하는 것이면 mode를 max로 지정
'''
mode : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준을 지정
- 예를 들어 관찰 항목이 ‘val_loss’인 경우에는 감소되는 것이 멈출 때 종료되어야 하므로, ‘min’으로 설정
auto : 관찰하는 이름에 따라 자동으로 지정합니다.
min : 관찰하고 있는 항목이 감소되는 것을 멈출 때 종료합니다.
max : 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료합니다.
'''

model.fit(x,y,epochs=10000,batch_size=1,verbose=2,
          callbacks=[es]) 


#4. 예측
x_input = x_input.reshape(1,3,1)
result = model.predict(x_input)
# print("x",x_input)
print("result : ",result)

loss = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)