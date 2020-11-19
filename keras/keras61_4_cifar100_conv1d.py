#Day9
#2020-11-19

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout

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
model = Sequential()
model.add(Conv1D(256, (3), padding="same", input_shape=(32, 3*32)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(128, (3), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(64, (3), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, (3), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# model.add(Conv1D(16, (4), padding="same"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(patience=50,mode='auto',monitor='val_loss')
modelpath = './model/cifar100_conv1d.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2, 
        validation_split=0.2, callbacks=[es,cp])

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/cifar100_conv1d.hdf5')


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
LSTM 모델
loss :  10.668805122375488
acc :  0.20819999277591705
예측값 :  [91 12  0 91 56 83  3 51 90 62]
실제값 :  [75 27 16 30 50 83 14 51 42 70]


Conv1D 모델
loss :  2.8943917751312256
acc :  0.2906999886035919
예측값 :  [75 49 62 30 69 92 21 18 74 70]
실제값 :  [75 27 16 30 50 83 14 51 42 70]
'''