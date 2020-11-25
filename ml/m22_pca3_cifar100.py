#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(50000,  32, 32, 3)

x = x.reshape(-1, 32*32*3) #reshape(-1, 정수)
# print(x.shape) #(60000, 3072)

#scaling
scaler = StandardScaler()
x = scaler.fit_transform(x) 

# PCA
pca = PCA(n_components=0.95) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
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
model.add(Dense(128, activation='relu', input_shape=(pca.n_components_,)) )
model.add(Dense(256, activation='relu') )
model.add(Dense(128, activation='relu') )
model.add(Dense(64, activation='relu') )
model.add(Dense(100, activation='softmax') )

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(patience=20,mode='auto',monitor='val_loss')
model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2, validation_split=0.2, callbacks=[es])

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
DNN without PCA*************
loss :  3.5389809608459473
acc :  0.20890000462532043
==================================================
PCA 0.95 : 3072 -> 202로 차원 축소 (PCA 후에 Standard Scaler)
loss :  6.0277018547058105
acc :  0.1662999987602234

PCA 0.95 : 3072 -> 207로 차원 축소 (PCA 전에 Standard Scaler)
loss :  5.947484493255615
acc :  0.18170000612735748
==================================================
PCA 0.99 : 3072 -> 661로 차원 축소 (PCA 후에 Standard Scaler)
loss :  10.00584888458252  
acc :  0.12049999833106995

PCA 0.99 : 3072 -> 661로 차원 축소 (PCA 전에 Standard Scaler)
loss :  8.285577774047852
acc :  0.16760000586509705
'''