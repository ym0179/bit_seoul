#Day3
#2020-11-11
#Multi-Layer Perceptron 다층 퍼셉트론

#1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array([range(101,201), range(311,411), range(100)])
print(x.shape) # (100, )
print(y.shape) # (3,100)
y = np.transpose(y)
print(y.shape) # (100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


#2. 모델 구성
# y1, y2, y3 = w1x1 + b
from tensorflow.keras.models import Sequential #순차적 모델 구성
from tensorflow.keras.layers import Dense #Dense model

model = Sequential()
# model.add(Dense(10, input_dim = 3))
model.add(Dense(10, input_shape = (1,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #출력 3개


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
print("R2 : ",r2)
