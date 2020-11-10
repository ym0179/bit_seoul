#Day2
#2020-11-10

'''
train_test_split 0.8 / train_valid split 0.7
loss:  [9.022187441587448e-10, 2.593994213384576e-05]
R2 :  0.9999999999991588

train_test_split 0.7 / train_valid split 0.7
loss:  [0.00944363884627819, 0.08250147849321365]
R2 :  0.9999901832930098

train_test_split 0.6 / train_valid split 0.7
loss:  [3.107804775238037, 1.503125786781311]
R2 :  0.9958677958004716

train_test_split 0.5 / train_valid split 0.7
loss:  [1221.2230224609375, 29.982032775878906]
R2 :  -0.34602627489879945

train_test_split 0.8 / train_valid split 0.5
loss:  [41.05054473876953, 5.494193077087402]
R2 :  0.906552511187276

train_test_split 0.8 / train_valid split 0.8
loss:  [1.800362952053547e-07, 0.0003562927304301411]
R2 :  0.9999999996982817

=> train / test split를 할 때 0.7 또는 0.8이 적당
'''

#1. 데이터
import numpy as np 
x = np.array(range(1,101)) # 1-100까지
y = np.array(range(101,201)) # 101-200까지

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7) #train 70%, test 30%
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3) #train 70%, test 30%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.5)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(3, input_dim = 1))
model.add(Dense(3, input_shape=(1,)))
# input_dim=1 = input_shape=(1,)
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
# print("결과: \n",y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ",r2) # max 값: 1
