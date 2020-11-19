#Day9
#2020-11-19

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout

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
model = Sequential()
# model.add(Conv1D(64, (3), padding="same", input_shape=(28,28)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.3))
model.add(Conv1D(32, (3), padding="same",input_shape=(28,28)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(16, (3), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(patience=10,mode='auto',monitor='val_loss')
modelpath = './model/mnist_conv1d.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2, 
        validation_split=0.2, callbacks=[es,cp])

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/mnist_conv1d.hdf5')


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
loss :  0.06250455975532532
acc :  0.9869999885559082
예측값 :  [7 8 9 0 1 2 3 4 5 6]
실제값 :  [7 8 9 0 1 2 3 4 5 6]

Conv1D 모델
loss :  0.05089378356933594
acc :  0.9860000014305115
예측값 :  [7 8 9 0 1 2 3 4 5 6]
실제값 :  [7 8 9 0 1 2 3 4 5 6]
'''