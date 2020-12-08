#Day22
#2020-12-08

#GPU 2개 돌리기 (분산처리)
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator?hl=ko

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
# print(y_train[0])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# print(x_train[0])

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf

#분산처리
strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
    tf.distribute.HierarchicalCopyAllReduce()
)

with strategy.scope(): #모델 - compile 까지 
    model = Sequential()
    model.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1))) #output: (28,28,10)
    model.add(Conv2D(30, (2,2), padding="same"))
    model.add(Conv2D(20, (2,2), padding="same"))
    model.add(Conv2D(15, (2,2), padding="same"))
    model.add(Conv2D(10, (2,2), padding="same"))
    model.add(Conv2D(5, (2,2), padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(patience=3,mode='auto',monitor='loss')
to_hist = TensorBoard(
    log_dir= "graph",
    histogram_freq=0,
    write_graph=True,
    write_images=True
)
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es,to_hist])

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
