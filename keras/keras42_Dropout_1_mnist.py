#Day6
#2020-11-16

#mnist (0-9까지의 손글씨) 예제
#Dropout: 신경망 학습을 할 때 전체가 아닌 일부의 뉴런만 사용

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding

#원핫인코딩을 하면 (60000,) => (60000,10)으로 reshape (분류 10개)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
print(y_train[0])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
print(x_train[0])

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1))) #output: (28,28,10)
model.add(Dropout(0.2)) #80%노드만 씀
model.add(Conv2D(30, (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(20, (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(15, (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(5, (2,2), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


model.summary()
# Dropout 가중치 연산으로 parameter#는 동일하게 나옴
# 연산이 빨라지고, 성능이 좋아짐, 과적합 잡아줌


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=3,mode='auto',monitor='loss')

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
without Dropout
loss :  0.1164519414305687
acc :  0.9800999760627747

with Dropout
loss :  0.35055333375930786
acc :  0.902400016784668
예측값 :  [7 8 4 0 1 2 3 4 5 6]
실제값 :  [7 8 9 0 1 2 3 4 5 6]
*Dropout이 항상 성능이 좋아지는 것은 아님. 어떤 layer에 적용? => hyperparameter 적용
'''