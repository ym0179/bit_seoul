#Day6
#2020-11-16

#mnist (0-9까지의 손글씨) 예제
#mnist를 DNN으로 코딩하시오
#(60000,28,28) -> (60000,784)

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
# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
# print(y_train[0])


# Deep Neural Network 모델의 input으로 넣기 위해 
# (28 by 28) 2차원 배열 (2D array) 이미지를 (28 * 28 = 784) 의 옆으로 길게 펼친 데이터 형태로 변형(Reshape)
x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train[0])

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
# model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(150, activation='relu',input_shape=(28*28,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(patience=5,mode='auto',monitor='loss')
# to_hist = TensorBoard(
#     log_dir= "graph",
#     histogram_freq=0,
#     write_graph=True,
#     write_images=True
# )
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
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

'''
CNN & DNN 비교
CNN
loss :  0.1164519414305687
acc :  0.9800999760627747

DNN
loss :  0.17194819450378418
acc :  0.9768000245094299
'''