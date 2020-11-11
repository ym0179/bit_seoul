#Day3
#2020-11-11

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array(range(101,201))
x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


#2. 모델 구성
#y = w1x1 + w2x2 + w3x3 + b
from tensorflow.keras.models import Sequential #순차적 모델 구성
from tensorflow.keras.layers import Dense #Dense model

model = Sequential()
# model.add(Dense(10, input_dim = 3))
model.add(Dense(10, input_shape = (3,))) #(100,3)
# (100,10,3) : input_shape=(10,3)
# 맨 앞은 행 = 전체 데이터의 갯수
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
model.fit(x_train,y_train,validation_split=0.2,batch_size=1,epochs=100)


#4. 평가,예측
loss, mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss: ",loss)
print("MAE: ",mae)

y_pred = model.predict(x_test)
# print("결과: \n",y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ",r2) # max 값: 1
