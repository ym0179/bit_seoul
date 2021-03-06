#Day20
#2020-12-04

from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape) #(8982,) (2246,) # 문장 8982개
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(x_train[0])
print(y_train[0])

#각 어절이 길이가 일정하지 않음 -> pad_sequence로 길이 맞춰줌
print(len(x_train[0])) #87
print(len(x_train[11])) #59

category = np.max(y_train) + 1
print("카테고리 : ", category) #46 -> y에 들어가는 카테고리 (신문기사 카테고리)

#y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo) #0부터 45

#y 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

#실습 : 모델 구성

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 1000
x_train = pad_sequences(x_train, padding='pre',maxlen=max_len) 
x_test = pad_sequences(x_test, padding='pre', maxlen=max_len) 
print(x_train.shape) 
print(x_test.shape) 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(10000,256)) #두가지 방식의 기법 #원핫인코딩 벡터화
model.add(LSTM(128))
model.add(Dense(46, activation = 'softmax')) #scala

# model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5)
model.fit(x_train, y_train, batch_size=64, epochs=100, callbacks=[es])

acc = model.evaluate(x_test, y_test)[1] #metrics 반환
print("acc : ", acc)

#acc :  0.699020504951477