#Day9
#2020-11-19


#just like the CIFAR-10, except it has 100 classes containing 600 images each 
#There are 500 training images and 100 testing images per class
#The 100 classes in the CIFAR-100 are grouped into 20 superclasses
#Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

x_train = x_train.reshape(50000, 32, 3*32).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 3*32).astype('float32')/255.

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
#input shape
#DNN - 1차원, RNN - 2차원, LSTM - 2차원
model = Sequential()
#lstm는 activation default tanh
#(행,열,몇개씩 자르는지) -> 마지막에 LSTM 만들 때 한개씩 잘라서 연산하겠다는게 명시됨 = 32개로 나눔
model.add(LSTM(64, activation='relu',input_shape=(32,3*32)))
# model.add(LSTM(62, activation='relu',return_sequences=True))
# model.add(LSTM(16, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#다중분류에서는 loss가 categorical crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=100,mode='auto',monitor='val_loss')
model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
loss :  10.668805122375488
acc :  0.20819999277591705
예측값 :  [91 12  0 91 56 83  3 51 90 62]
실제값 :  [75 27 16 30 50 83 14 51 42 70]

'''