#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(50000,  32, 32, 3)

x = x.reshape(-1, 32*32*3) #reshape(-1, 정수)
# print(x.shape) #(60000, 3072)

#scaling
scaler = StandardScaler()
x = scaler.fit_transform(x) 

# PCA
pca = PCA(n_components=0.99) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
x = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)

#train test 다시 나누기
x_train = x[:50000,:]
x_test = x[-10000:,:]

# print(x_train.shape)
# print(x_test.shape)

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
model.add(Dense(64, activation='relu',input_shape=(pca.n_components_,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(patience=10,mode='auto',monitor='loss')
model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=1, validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

result = model.predict(x_pred)

# argmax는 가장 큰 값의 인덱스 값을 반환
y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
DNN without PCA
loss :  2.0349631309509277
acc :  0.4302999973297119
===============================================================
PCA 0.95 : 3072 -> 217로 차원 축소 (PCA 후에 Standard Scaler
loss :  1.4575376510620117
acc :  0.4677000045776367

PCA 0.95 : 3072 -> 221로 차원 축소 (PCA 전, 후에 Standard Scaler)
loss :  1.4540990591049194
acc :  0.47440001368522644

PCA 0.95 : 3072 -> 221로 차원 축소 (PCA 전에 Standard Scaler) ************
loss :  1.4311307668685913
acc :  0.4893999993801117
===============================================================
PCA 0.99 : 3072 -> 660로 차원 축소 (PCA 후에 Standard Scaler
loss :  1.6835248470306396
acc :  0.41029998660087585

PCA 0.99 : 3072 -> 664로 차원 축소 (PCA 전, 후에 Standard Scaler)
loss :  1.799400806427002
acc :  0.41290000081062317

PCA 0.99 : 3072 -> 664로 차원 축소 (PCA 전에 Standard Scaler)
loss :  1.5768656730651855
acc :  0.45989999175071716
'''