#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# mnist dnn과 loss/acc 비교

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(506, 13) (506,)

#scaling
scaler = StandardScaler()
x = scaler.fit_transform(x) 

# PCA
pca = PCA(n_components=0.99) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
x = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)

#train test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7)

#scaling
# scaler = StandardScaler()
# scaler.fit(x_train) #fit은 train data만 함
# x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
# x_test = scaler.transform(x_test)

#pred 만들기
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(pca.n_components_,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 

#4. 평가, 예측
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

result = model.predict(x_pred)
print("예측값 : ", result.T.reshape(10,)) #보기 쉽게
print("실제값 : ", y_pred)

y_predicted =  model.predict(x_test) #x_pred 10개밖에 없음응로 x_test 가지고 RMSE, R2 계산

#RMSE
#R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predicted):
    return np.sqrt(mean_squared_error(y_test,y_predicted))
print("RMSE : ", RMSE(y_test, y_predicted))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predicted)
print("R2 : ",r2) # max 값: 1

'''
DNN without PCA
loss :  1.2785238027572632
mae :  0.8258237838745117      
RMSE :  1.130718411993961
R2 :  0.9846784372162021
=====================================
PCA 0.95 : 13 -> 2로 차원 축소 (PCA 후에 Standard Scaler)
loss :  50.63014602661133
mae :  5.042294979095459
RMSE :  7.115486722205497
R2 :  0.3415764517145592

PCA 0.95 : 13 -> 9로 차원 축소 (PCA 전, 후에 Standard Scaler)
loss :  5.708744525909424
mae :  1.8087358474731445
RMSE :  2.3892978328522934
R2 :  0.930966345307046

PCA 0.95 : 13 -> 9로 차원 축소 (PCA 전에 Standard Scaler)
loss :  2.9149980545043945
mae :  1.254248023033142
RMSE :  1.7073363910226609
R2 :  0.9659450396427731
======================================
PCA 0.99 : 13 -> 3로 차원 축소 (PCA 후에 Standard Scaler)
loss :  51.04422378540039
mae :  4.786080837249756
RMSE :  7.144524939884035
R2 :  0.41866744375359644

PCA 0.99 : 13 -> 12로 차원 축소 (PCA 전, 후에 Standard Scaler) **********
loss :  1.4658678770065308
mae :  0.8694745898246765
RMSE :  1.2107302329384735
R2 :  0.9805799081355376

PCA 0.99 : 13 -> 12로 차원 축소 (PCA 전에 Standard Scaler)
loss :  1.5523089170455933
mae :  0.9556472301483154
RMSE :  1.245916790062786
R2 :  0.9793385625688545
'''