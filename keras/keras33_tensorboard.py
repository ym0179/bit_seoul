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
# print("x.shape:", x.shape)
x = x.reshape(x.shape[0], x.shape[1], 1) # (4,3,1)
# print("reshape x.shape:", x.shape)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7)
x_test ,x_val, y_test, y_val = train_test_split(
    x_train, y_train, train_size=0.7)

#모델을 구성하시오.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# model = Sequential()
# model.add(LSTM(75, activation='relu', input_length=4, input_dim=1))
# model.add(Dense(180, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(110, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1)) #output: 1개
input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(75, activation='relu')(input1)
dense = Dense(180, activation='relu')(lstm)
dense = Dense(150, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard  # 텐서보드 = 웹페이지 시각화

es = EarlyStopping(monitor='loss', patience=15)

to_hist = TensorBoard(log_dir="graph", #log가 들어갈 폴더 지정 -> graph라는 폴더 만들어 줘야함
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

history = model.fit(
    x_train,
    y_train,
    callbacks=[es,to_hist],
    validation_data=(x_val,y_val),
    epochs=800, 
    verbose=2
)
'''
1.cmd창 켜기
2.graph 폴더로 들어가서 (cd)
3.명령어 tensorboard --logdir=. 
4.http://localhost:6006/ - 텐서보드 웹
'''
# # 그래프
# import matplotlib.pyplot as plt

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.plot(history.history["mae"])
# plt.plot(history.history["val_mae"])

# plt.title("loss & mae")
# plt.ylabel("loss, mae")
# plt.xlabel("epochs")

# plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
# plt.show()

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss) # 이건 기본으로 나오고
print("mae: ", mae)

# x_predict = np.array([97, 98, 99, 100])
# x_predict = x_predict.reshape(1, 4, 1)

# y_predict = model.predict(x_predict)
# print("예측값: ", y_predict)