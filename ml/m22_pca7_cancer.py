#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(569, 30) (569,)

# PCA
pca = PCA(n_components=0.95) #데이터셋에 분산의 n%만 유지하도록 PCA를 적용
x = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)

#train test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7)

#scaling
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#pred 만들기
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(pca.n_components_,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='loss',patience=10,mode='auto')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 

#4. 평가, 예측
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

result = model.predict(x_pred)

print("예측값 : ", result.T.reshape(10,))
print("실제값 : ", y_pred)

'''
DNN without PCA
loss :  0.5096859931945801
acc :  0.9824561476707458

PCA 0.95 : 30 -> 1로 차원 축소
loss :  0.21957197785377502
acc :  0.9213836193084717

PCA 0.99 : 30 -> 2로 차원 축소
loss :  0.11446408182382584
acc :  0.9496855139732361
'''