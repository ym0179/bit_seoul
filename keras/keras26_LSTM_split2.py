#Day5
#2020-11-13


#실습: 모델 구성
#train, test 분리하기 + early_stopping + validation_split
#predict


import numpy as np

#1. 데이터
dataset = np.array(range(1,101))
size = 5

#데이터 전처리 
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)

    # print(type(aaa))
    return np.array(aaa)


datasets = split_x(dataset, size)

x = datasets[:, :4]
y = datasets[:, 4]
x = x.reshape(x.shape[0], x.shape[1], 1)

#차원과는 관계없이 비례에 맞춰서 잘라 준다!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)

#모델을 구성하시오.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer
model = Sequential()
model.add(LSTM(30, activation='relu', input_length=4, input_dim=1)) # *****
model.add(Dense(50, activation='relu')) #default activation = linear
model.add(Dense(70, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1)) #output: 1개


#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.3,
    epochs=1000, batch_size=10
)


#4. 평가, 예측

#101.0
loss, mse = model.evaluate(x_test, y_test)
print(loss, mse)


x_predict = np.array([97, 98, 99, 100])
x_predict = x_predict.reshape(1, 4, 1)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)