#Day2
#2020-11-10
#복습

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

#train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.7)

# print(x_train)
# print(x_val)
# print(x_test)

#2.모델 구성
model = Sequential()
model.add(Dense(5,input_shape=(1,)))
model.add(Dense(3))
model.add(Dense(7))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=100,batch_size=1)

#4.평가, 예측
loss = model.evaluate(x_test,y_test,batch_size=1)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print("RMSE : ",RMSE(y_test,y_pred))
print("r2 : ",r2)