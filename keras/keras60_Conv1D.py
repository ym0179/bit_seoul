#Day9
#2020-11-19

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D

# 1. 데이터
a = np.array(range(1,101)) #1부터 99
print(a)
size = 5

# split_x 함수 데려오기
def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)        
    return np.array(aaa)

datasets = split_x(a, size)

x = datasets[:, :4]
y = datasets[:, 4]
print(x.shape) #(95, 4)
print(y.shape) #(95,)
# print(x)
# print(y)
x_pred = x[-1:]
# print(x_pred)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)


# conv1D로 모델 구성하시오
# 2. 모델링
model = Sequential()
model.add(Conv1D(16, (2), padding="same", input_shape=(4,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, (2), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(patience=10,monitor='val_loss')

modelpath = './model/conv1d.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

model.fit(x_train,y_train,epochs=100,callbacks=[es,cp],
        batch_size=1,validation_split=0.2,verbose=2)


# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/conv1d.hdf5')


# 4. 평가
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)


# 5. 예측
result = model.predict(x_pred)
print('predicted : ', result.reshape(1,))

y_predicted = model.predict(x_test)

#RMSE
#R2
import numpy as np
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predicted):
    return np.sqrt(mean_squared_error(y_test,y_predicted))
print("RMSE : ", RMSE(y_test, y_predicted))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predicted)
print("R2 : ",r2) # max 값: 1

'''
loss :  0.002083364874124527
mae :  0.035810042172670364
predicted :  [99.98238]
RMSE :  0.04564398859049918
R2 :  0.99999759325621
'''