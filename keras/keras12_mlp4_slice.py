#Day3
#2020-11-11
#실습 train_test_split를 슬라이싱으로 바꿀 것


#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array([range(101,201), range(311,411), range(100)])
print(x.shape) # (3,100)
x = np.transpose(x)
y = np.transpose(y)
print(x.shape) # (100,3)

#slicing
x_train = x[:60]
y_train = y[:60]
x_val = x[60:80]
y_val = y[60:80]
x_test = x[-20:]
y_test = y[-20:]

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.7)


#2. 모델 구성
# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim = 3)) 
model.add(Dense(10, input_shape = (3,))) 
model.add(Dense(5))
model.add(Dense(3)) #출력 3개


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
# model.fit(x_train,y_train,validation_split=0.2,batch_size=1,epochs=100)
model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=1,epochs=100)


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
