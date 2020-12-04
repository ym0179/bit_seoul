#Day20
#2020-12-04

#최적화 튠으로 구성

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


#2. 모델
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (3,3), padding="same"))
model.add(Conv2D(128, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
cp = ModelCheckpoint('./model/keras77_cifar10.hdf5', monitor='val_loss', 
                    save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', 
                       patience=3,  # epoch 3 동안 개선되지 않으면 callback이 호출
                       factor=0.5 # callback 호출시 학습률을 1/2로 줄임
                       ) 

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
          validation_split=0.2, callbacks=[cp, es, rl])

model = load_model('./model/keras77_cifar10.hdf5')


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", loss)
print("acc: ", acc)

'''
loss:  0.9148290157318115
acc:  0.7113000154495239
'''