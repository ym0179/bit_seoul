#Day7
#2020-11-17

#CIFAR-10 dataset은 32x32픽셀의 60000개 컬러이미지가 포함되어있으며, 각 이미지는 10개의 클래스로 라벨링

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

#predict 만들기
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
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
# model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#다중분류에서는 loss가 categorical crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=30,mode='auto',monitor='loss')
model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=2, validation_split=0.2, callbacks=[es])

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
loss :  1.434055209159851
acc :  0.7010999917984009
예측값 :  [3 8 8 8 4 6 6 3 3 1]
실제값 :  [3 8 8 0 6 6 1 6 3 1]
'''