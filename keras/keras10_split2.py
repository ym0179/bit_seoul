#Day2
#2020-11-10

#1. 데이터
import numpy as np 
x = np.array(range(1,101)) # 1-100까지
y = np.array(range(101,201)) # 101-200까지

x_train = x[:60] #60개
y_train = y[:60]
x_val = x[60:80] #20개
y_val = x[60:80]
x_test = x[-20:] #20개
y_test = y[-20:]

# print(x_train)
# print(x_val)
# print(x_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
# model.fit(x_train,y_train,validation_split=0.2,batch_size=1,epochs=100)
model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=1,epochs=100)


#4. 평가,예측
loss = model.evaluate(x_test,y_test,batch_size=1)
print("loss: ",loss)
y_pred = model.predict(x_test)
print("결과: \n",y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ",r2) # max 값: 1
