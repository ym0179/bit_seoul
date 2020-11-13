#Day5
#2020-11-13


#1. 데이터
import numpy as np
dataset = np.array(range(1,101))
size = 5

#데이터 전처리 
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    return np.array(aaa)

datasets = split_x(dataset, size)

x = datasets[:, :4]
y = datasets[:, 4]

# x = x.reshape(x.shape[0], 4, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=False #보기 편하게 하기 위해 - 성능을 원하면: True
)

#모델을 구성하시오.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
model = Sequential()
model.add(LSTM(30, activation='relu', input_length=4, input_dim=1))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1)) #output: 1개


#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=15, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(
    x_train,
    y_train,
    callbacks=[es],
    validation_split=0.2,
    epochs=1000, batch_size=10
)

print("========================")
print(history) 
#history 안에 자료형 반환함 (i.e <tensorflow.python.keras.callbacks.History object at 0x000002178F548B50>)

print("========================")
print(history.history.keys()) #history.keys()
#loss,metrics,val_loss,val_metrics 나옴
#dict_keys(['loss', 'mse', 'val_loss', 'val_mse'])

print("========================")
print(history.history['loss']) #dict 반환되는 4개중에 loss 인거
#epochs당 하나씩 생성된 loss값 print

print("========================")
print(history.history['val_loss']) #dict 반환되는 4개중에 val_loss 인거
#epochs당 하나씩 생성된 val_loss값 print

'''
model.fit에서 history로 dict 반환
loss, metrics 값 epochs 별로 확인 가능
'''

# #4. 평가, 예측
# loss, mse = model.evaluate(x_test, y_test)
# print("lose : ",loss, mse)

# x_predict = np.array([97, 98, 99, 100])
# x_predict = x_predict.reshape(1, 4, 1)

# y_predict = model.predict(x_predict)
# print("예측값: ", y_predict)