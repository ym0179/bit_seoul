#Day8
#2020-11-18

#다중분류
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

#1. 전처리

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#reshape
x_train = x_train.reshape(x_train.shape[0], 4, 1)
x_val = x_val.reshape(x_val.shape[0], 4, 1)
x_test = x_test.reshape(x_test.shape[0], 4, 1)

#predict 만들기 - train-test 에서 shuffle 하고 나서 해줌
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
#input shape
#DNN - 1차원, RNN - 2차원, LSTM - 2차원
model = Sequential()
#(행,열,몇개씩 자르는지) -> 마지막에 LSTM 만들 때 한개씩 잘라서 연산하겠다는게 명시됨
model.add(LSTM(64, activation='relu',input_shape=(4,1)))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10,mode='auto')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 


#4. 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

#5. 예측
result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)
'''
loss :  0.15735237300395966
acc :  0.9666666388511658
예측값 :  [2 2 1 2 2 0 0 0 1 0]
실제값 :  [2 2 1 2 2 0 0 0 2 0]
'''

