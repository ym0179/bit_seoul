#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(150, 4) (150,)

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# PCA
pca = PCA(n_components=0.99) #데이터셋에 분산의 n%만 유지하도록 PCA를 적용
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
# model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='loss',patience=10,mode='auto')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 

#4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
DNN without PCA *********
loss :  0.03717958182096481
acc :  1.0
=============================================================
PCA 0.95 : 4 -> 2로 차원 축소 (PCA 후에 Standard Scaler)
loss :  0.03865565359592438
acc :  0.988095223903656
=============================================================
PCA 0.99 : 4 -> 3로 차원 축소 (PCA 후에 Standard Scaler)
loss :  0.0
acc :  1.0
'''