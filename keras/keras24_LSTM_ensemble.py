#Day4
#2020-11-12

#1. 데이터
import numpy as np 
from numpy import array
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120], 
              [2,3,4], [3,4,5], [4,5,6]]) #(13,3)
x1 = x1.reshape(13,3,1)
x2 = x2.reshape(13,3,1)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13,)
x1_predict = array([55,65,75]) #(3, )
x2_predict = array([65,75,85])#(3, )
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)


##### 실습 : 앙상블 모델을 만드시오.
#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

input1 = Input(shape=(3,1))
lstm1 = LSTM(30,activation='relu')(input1)
dense1 = Dense(15,activation='relu')(lstm1)
dense1 = Dense(10,activation='relu')(dense1)
output1 = Dense(7,activation='relu')(dense1)
output1 = Dense(3,activation='relu')(dense1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(20,activation='relu')(input2)
dense2 = Dense(10,activation='relu')(lstm2)
dense2 = Dense(7,activation='relu')(dense2)
output2 = Dense(5,activation='relu')(dense2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1,output2])
middle1 = Dense(20,activation='relu')(merge1)
# middle1 = Dense(20,activation='relu')(middle1)
middle1 = Dense(15,activation='relu')(middle1)
# middle1 = Dense(10,activation='relu')(middle1)

output = Dense(15,activation='relu')(middle1)
output = Dense(7,activation='relu')(output)
output = Dense(5,activation='relu')(output)
output = Dense(1)(output)

model = Model(inputs=[input1,input2],outputs=output)

#3. 컴파일, 훈련
model.compile(loss="mse",optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=15,mode='auto')
model.fit([x1,x2],y,epochs=1000,batch_size=1,verbose=2,callbacks=[es])

#4. 평가, 예측
loss = model.evaluate([x1,x2],y,batch_size=1)
print("loss : ",loss)
result1 = model.predict([x1_predict, x2_predict])
result2 = model.predict([x2_predict, x1_predict])
print("result1 : ",result1) #85
print("result2 : ",result2) #95

