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

# 실습 GRU 완성하시오
# 예측값 80

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
model = Sequential()
model.add(GRU(35,activation='relu',input_shape=(3,1)))
model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

model.compile(loss="mse",optimizer='adam')
model.fit(x,y,epochs=700,batch_size=1,verbose=2) 


#4. 예측
x_input = x_input.reshape(1,3,1)
result = model.predict(x_input)
print("result : ",result)

loss = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)

'''
Layer (type)                 Output Shape        
===============================================      
LSTM / SimpleRNN             (None, 35)                
_______________________________________________    
dense (Dense)                (None, 20)          
_______________________________________________        
dense_1 (Dense)              (None, 15)         
_______________________________________________        
dense_2 (Dense)              (None, 7)            
_______________________________________________        
dense_3 (Dense)              (None, 1)               
===============================================  

SimmpleRNN 
예측값(80): 80.13006
loss: 0.0002845380222424865
param #: 1295 = 1*(1+1+35)*35
    - model.add(SimpleRNN(35, input_shape=(3,1)))

LSTM
예측값(80): 80.65509
loss:  0.03869479522109032
param #: 5180 = 4*(1+1+35)*35
    - model.add(LSTM(35, input_shape=(3,1)))

GRU
예측값(80): 80.8288680.82886
loss: 0.0006416416144929826
param #: 3990 = 3*(1+1+35+1)*35
    - model.add(GRU(35, input_shape=(3,1)))
'''