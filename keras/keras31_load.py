#Day5
#2020-11-13

#모델 로드

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
from tensorflow.keras.models import load_model,Model #Sequential 없이도 돌아감! load_model에서 같이 당겨온다 
from tensorflow.keras.layers import Dense, LSTM, Input #LSTM도 layer

# 모델 불러오기
model = load_model('./save/keras26_model.h5')
model.summary()

#input layer 추가
# model.add(LSTM(30, activation='relu', input_length=4, input_dim=1, name="input_layer"))

# 함수형
# input1 = Input(shape=(4,1), name='input1')
# dense = model(input1)
# output1 = Dense(1)(dense)
# model = Model(inputs=input1, outputs=output1)
# model.summary()

# 시퀀셜로도 가능함
model(Input(shape=(4,1), name='input1'))
model.add(Dense(1, name='output1'))
model.summary()


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
#101.03
loss, mse = model.evaluate(x_test, y_test)
print(loss, mse)


x_predict = np.array([97, 98, 99, 100])
x_predict = x_predict.reshape(1, 4, 1)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)