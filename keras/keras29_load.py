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


dataset = split_x(dataset, size)


#dataset[:, 0:4]
#dataset[:, 4]
#shape 확인하고 print한 다음 주석으로 적어 두기 

x = dataset[0:100, 0:4]
y = dataset[0:100, 4]


x = x.reshape(x.shape[0], 4, 1)

#차원과는 관계없이 비례에 맞춰서 잘라 준다!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)


#모델을 구성하시오.
from tensorflow.keras.models import load_model #Sequential 없이도 돌아감! load_model에서 같이 당겨온다 
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer


# model = Sequential()
# model.add(LSTM(100, input_shape=(4, 1)))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

# model.add(Dense(1)) #output: 1개



# 모델 불러오기
model = load_model('./save/keras26_model.h5')


#기존 모델에 커스터마이징하기
# ValueError: All layers added to a Sequential model should have unique names. 
# Name "dense" is already the name of a layer in this model. 
# Update the `name` argument to pass a unique name.    
# name 지정

# model.add(Dense(10, name="king1"))
model.add(Dense(1, name="king2"))


# model.summary() #커스터마이징 할 때마다 돌려서 코드 새로 쓸 순 x




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    x,
    y,
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