#Day7
#2020-11-17

#보스턴 집값 예측: 1978년에 발표된 데이터로 미국 보스턴 지역의 주택 가격에 영향을 미치는 요소들을 정리


from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(506, 13) (506,)

x_pred = x[:10]
y_pred = y[:10]

#1. 전처리
#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x) #fit은 train data만 함
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

#reshape
x = x.reshape(506,13,1,1)
x_pred = x_pred.reshape(10,13,1,1)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(32, (1,1), padding="same", input_shape=(13,1,1)))
model.add(Conv2D(16, (1,1), padding="same"))
model.add(Conv2D(8, (1,1), padding="same"))
# model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x,y,epochs=300,batch_size=1,verbose=2,callbacks=[es]) 


#4. 평가
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
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
loss :  14.791457176208496
mae :  2.8353798389434814
예측값 :  [30.477022 23.248276 34.868256 33.00118  32.658356 25.660847 20.541096
RMSE :  3.8459666794160223
R2 :  0.8460726474462912
'''

