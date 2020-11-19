#Day9
#2020-11-19


from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(506, 13) (506,)

#1. 전처리
#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#reshape
x_train = x_train.reshape(x_train.shape[0],13,1)
x_val = x_val.reshape(x_val.shape[0],13,1)
x_test = x_test.reshape(x_test.shape[0],13,1)


x_pred = x_test[:10]
y_pred = y_test[:10]

#2. 모델링
model = Sequential()
model.add(Conv1D(128, (2), padding="same", input_shape=(13,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(64, (2), padding="same", input_shape=(13,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(32, (2), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
modelpath = './model/boston_conv1d.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train,y_train,epochs=400,batch_size=1,verbose=2,callbacks=[es,cp], validation_data=(x_val,y_val)) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/boston_conv1d.hdf5')


#4. 평가
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
result = model.predict(x_pred)
print("예측값 : ", result.T.reshape(10,)) #보기 쉽게
print("실제값 : ", y_pred)
y_predicted =  model.predict(x_test) #x_pred 10개밖에 없음응로 x_test 가지고 RMSE, R2 계산

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
LSTM 모델
loss :  12.263466835021973
mae :  2.7167487144470215
예측값 :  [25.90948    6.2764387 20.263472  17.902828  13.495611  26.259878
 19.45948   22.261282  23.709982  23.103811 ]
실제값 :  [23.1 10.4 17.4 20.5 13.  20.5 21.8 21.2 21.8 23.1]
RMSE :  3.5019234178103877
R2 :  0.8028192283008149

Conv1D 모델
loss :  10.494782447814941
mae :  2.511798143386841
예측값 :  [33.461502 21.24094  14.474295 20.45633  20.079025 10.418794 20.638466
 10.255644 19.675922 16.402971]
실제값 :  [34.7 21.5 17.9 24.3 17.5 11.5 19.9 13.4 18.  13.5]       
RMSE :  3.239564889289021
R2 :  0.8928022270130935
'''


