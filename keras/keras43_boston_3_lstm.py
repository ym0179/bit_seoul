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

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# model.add(Dense(32, activation='relu',input_shape=(13,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))


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
#1
RMSE :  1.368949050545954
R2 :  0.9734144756232456

#2
예측값 :  [24.692533 21.921623 33.017033 32.04224  32.26689  25.371145 20.51319   
 18.475174 14.318202 18.430988]
실제값 :  [24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9]
RMSE :  1.8120813885400564
R2 :  0.9644983026255848
'''