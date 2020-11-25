#Day13
#2020-11-25

# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 0.99이상
# dnn과 loss/acc 비교

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(506, 13) (506,)

# PCA
pca = PCA(n_components=0.95) #데이터셋에 분산의 n%만 유지하도록 PCA를 적용
x = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)

#train test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=44)

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
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

es = EarlyStopping(monitor='val_loss',patience=70,mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es],validation_split=0.3) 

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
import numpy as np
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predicted):
    return np.sqrt(mean_squared_error(y_test,y_predicted))
print("RMSE : ", RMSE(y_test, y_predicted))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predicted)
print("R2 : ",r2) # max 값: 1

'''
DNN without PCA
loss :  1523.615478515625
mae :  27.910646438598633
RMSE :  39.03351252066182
R2 :  0.7501489989808323

PCA 0.95 : 13 -> 8로 차원 축소
loss :  1717.641357421875
mae :  31.407352447509766
RMSE :  41.44443601238127
R2 :  0.7183314917390959


PCA 0.99 : 13 ->8로 차원 축소
loss :  1998.182373046875
mae :  33.25697326660156
RMSE :  44.701029686482464
R2 :  0.6723268319102942
'''