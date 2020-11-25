#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1 이상
# mnist dnn과 loss/acc 비교

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)

x = x.reshape(-1, 28*28) #reshape(-1, 정수)
# print(x.shape) #(70000, 784)

# PCA
pca = PCA(n_components=713) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
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
model.add(Dense(150, activation='relu',input_shape=(713,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(patience=5,mode='auto',monitor='loss')
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2, validation_split=0.3, callbacks=[es])

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
DNN without PCA
loss :  0.17194819450378418
acc :  0.9768000245094299

PCA 0.95 : 784 -> 154로 차원 축소
loss :  0.19751688838005066 (Standard Scaler)
acc :  0.9660000205039978

loss :  0.21488747000694275 (Scaler X)
acc :  0.9700999855995178

PCA 1.0 : 784 -> 713로 차원 축소
loss :  0.3176426291465759 (Scaler X)
acc :  0.9657999873161316

loss :  0.5177420973777771 (Standard Scaler)
acc :  0.9369999766349792

loss :  0.625217080116272 (MinMax Scaler)
acc :  0.7958999872207642
'''