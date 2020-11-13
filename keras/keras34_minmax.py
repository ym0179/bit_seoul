#Day5
#2020-11-13

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
           [100,200,300] #(14,3)
        ])
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])
x_pred = array([55,65,75])  #(3,)
x_pred = x_pred.reshape(1,3)   #(1,3)
x_pred2 = array([6600,6700,6800])  #(3,)


# 데이터 전처리 - scaling
# x값만 하면됨
# MinMaxScaler: 모든 feature가 0과 1사이에 위치하게 만듬
# Xscale = (X-Xmin) / (Xmax-Xmin)
from sklearn.preprocessing import MinMaxScaler #2차원 까지
scaler = MinMaxScaler()
scaler.fit(x) #fit은 train data만 함
x = scaler.transform(x) #0과 1사이의 값
x_pred = scaler.transform(x_pred)
# x = scaler.fit_transform(x)
# print(x)

# x = x.reshape(x.shape[0], x.shape[1], 1) #(14,3,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(
    x_train, y_train, train_size=0.7)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# input1 = Input(shape=(x.shape[1],1))
# lstm = LSTM(20, activation='relu')(input1)
# dense = Dense(10, activation='relu')(lstm)
# dense = Dense(7, activation='relu')(dense)
# output1 = Dense(1)(dense)
# model = Model(inputs=input1, outputs=output1)
model = Sequential()
model.add(Dense(40, activation = 'relu', input_shape = (3,)))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))



#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='loss', patience=10)

to_hist = TensorBoard(log_dir="graph",
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

history = model.fit(
    x_train,
    y_train,
    callbacks=[es,to_hist],
    validation_data=(x_val,y_val),
    epochs=500, 
    verbose=2,
    batch_size=1
)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_pred = model.predict(x_pred)
print("예측값: ", y_pred)