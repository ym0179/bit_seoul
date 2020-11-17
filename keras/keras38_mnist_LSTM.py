#Day7
#2020-11-17

#mnist (0-9까지의 손글씨) 예제
#mnist를 LSTM으로 코딩하시오

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding

#원핫인코딩을 하면 (60000,) => (60000,10)으로 reshape (분류 10개)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
# print(y_train[0])

# x_train = x_train.reshape(60000, 784, 1).astype('float32')/255. #28*28
# x_test = x_test.reshape(10000, 784, 1).astype('float32')/255.
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#input shape
#DNN - 1차원, RNN - 2차원, LSTM - 2차원
model = Sequential()
#lstm는 activation default tanh
#(행,열,몇개씩 자르는지) -> 마지막에 LSTM 만들 때 한개씩 잘라서 연산하겠다는게 명시됨 = 28개로 나눔
model.add(LSTM(128, activation='relu',input_shape=(28,28),return_sequences=True))
model.add(LSTM(64, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=5,mode='auto',monitor='loss')

model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

# print("x_pred : ", x_pred)
# print("y_pred : ", y_pred)

result = model.predict(x_pred)

# argmax는 가장 큰 값의 인덱스 값을 반환
y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)
