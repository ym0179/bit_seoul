#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(60000, 28, 28)

x = x.reshape(-1, 28*28) #reshape(-1, 정수)
# print(x.shape) #(60000, 784)

#scaling
scaler = StandardScaler()
x = scaler.fit_transform(x) 

# PCA
pca = PCA(n_components=0.95) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
x = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)

#train test 다시 나누기
x_train = x[:60000,:]
x_test = x[-10000:,:]

print(x_train.shape)
print(x_test.shape)

#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#scaling
# scaler = StandardScaler()
# scaler.fit(x_train) #fit은 train data만 함
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]

#2. 모델
model = Sequential()
model.add(Dense(128, activation='relu',input_shape=(pca.n_components_,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(patience=5,mode='auto',monitor='loss')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

result = model.predict(x_pred)

# argmax는 가장 큰 값의 인덱스 값을 반환
y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
DNN without PCA **********
loss :  0.8847358226776123
acc :  0.8852999806404114
===========================================================================
PCA 0.95 : 784 -> 188로 차원 축소 (PCA 후에 Standard Scaler)
loss :  1.2857699394226074
acc :  0.8586999773979187

PCA 0.95 : 784 -> 256로 차원 축소 (PCA 전, 후에 Standard Scaler)
loss :  1.2820942401885986
acc :  0.859000027179718

PCA 0.99 : 784 -> 527로 차원 축소 (PCA 전에 Standard Scaler)
loss :  1.411585807800293
acc :  0.8708000183105469
===========================================================================
PCA 0.99 : 784 -> 459로 차원 축소 (PCA 후에 Standard Scaler)
loss :  1.600558876991272
acc :  0.8508999943733215

PCA 0.99 : 784 -> 527로 차원 축소 (PCA 전, 후에 Standard Scaler)
loss :  1.5600082874298096
acc :  0.8514999747276306

PCA 0.99 : 784 -> 527로 차원 축소 (PCA 전에 Standard Scaler)
loss :  1.1956037282943726
acc :  0.8740000128746033
'''