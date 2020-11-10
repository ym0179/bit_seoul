#1. 데이터
import numpy as np 
x = np.array(range(1,101)) # 1-100까지
y = np.array(range(101,201)) # 101-200까지

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7) #train 70%, test 30%
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3) #train 70%, test 30%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.7)


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
# model.fit(x_train,y_train,batch_size=1,epochs=100)
model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=1,epochs=100)



#4. 평가,예측
loss = model.evaluate(x_test,y_test,batch_size=1)
print("loss: ",loss)
y_pred = model.predict(x_test)
print("결과: \n",y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ",r2) # max 값: 1
