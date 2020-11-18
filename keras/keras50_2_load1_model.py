#Day8
#2020-11-18

#mnist (0-9까지의 손글씨) 예제

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
print(y_train[0])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# #conv는 activation default relu
# #lstm는 activation default tanh
# #dense는 activation default linear
# model = Sequential()
# model.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1))) #output: (28,28,10)
# model.add(Conv2D(30, (2,2), padding="same"))
# model.add(Conv2D(20, (2,2), padding="same"))
# model.add(Conv2D(15, (2,2), padding="same"))
# model.add(Conv2D(10, (2,2), padding="same"))
# model.add(Conv2D(5, (2,2), padding="same"))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='softmax'))


#################### 모델 불러오기 #######################
from tensorflow.keras.models import load_model
model = load_model('./save/model_test01_1.h5') #compile & fit 전에 저장한 모델
model.summary()
'''
원래 모델 값:
loss :  0.09943143278360367
acc :  0.9761999845504761
예측값 :  [7 8 9 0 1 2 3 4 5 6]
실제값 :  [7 8 9 0 1 2 3 4 5 6]

model_test01_1.h5: 일치 안함 => compile, fit 전에 불러온 모델
loss :  0.10792809724807739
acc :  0.9767000079154968
예측값 :  [7 8 9 0 1 2 3 4 5 6]
실제값 :  [7 8 9 0 1 2 3 4 5 6]
'''


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(patience=10,mode='auto',monitor='val_loss')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, 
                 validation_split=0.2, callbacks=[es,cp])


#4. 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

predicted = model.predict(x_pred)

# argmax는 가장 큰 값의 인덱스 값을 반환
y_predicted = np.argmax(predicted,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)
