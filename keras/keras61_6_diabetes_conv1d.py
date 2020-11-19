#Day9
#2020-11-19

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(442, 10) (442,)
# print(dataset)

#1. 전처리
#train-test split -> scaling train 만 하니까 젤 먼저 split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

#scaling - 2차원 input만 가능 -> reshape 나중에
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#reshape
x_train = x_train.reshape(x_train.shape[0],10,1)
x_val = x_val.reshape(x_val.shape[0],10,1)
x_test = x_test.reshape(x_test.shape[0],10,1)

x_pred = x_test[:10]
y_pred = y_test[:10]

#2. 모델링
model = Sequential()
model.add(Conv1D(128, (2), padding="same", input_shape=(10,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(64, (2), padding="same"))
model.add(Dropout(0.3))
model.add(Conv1D(32, (2), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(32, (2), padding="same"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=100,mode='auto')
modelpath = './model/diabetes_conv1d-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train,y_train,epochs=1000,batch_size=16,verbose=2,
        callbacks=[es,cp],validation_data=(x_val,y_val)) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/diabetes_conv1d-57-3073.5078.hdf5')


#4. 평가
loss,mae = model.evaluate(x_test,y_test,batch_size=16)
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
loss :  3205.79638671875
mae :  44.46400833129883
예측값 :  [ 84.67883   68.966324 168.0709    66.70339  196.6143   191.96341
 102.08669  212.71727  120.2652   138.64984 ]
실제값 :  [113.  90. 232.  55. 246. 202. 111. 281.  59. 121.]
RMSE :  56.619756603559715
R2 :  0.4726743055910535

Conv1D 모델
loss :  2594.342529296875
mae :  40.9542121887207
예측값 :  [236.53426  109.184044  75.73746  192.03503  149.2699   178.84619
 243.17163  137.77966  175.20113   84.063515]
실제값 :  [281.  78.  59. 121. 129.  81. 268. 102. 144.  83.]       
RMSE :  50.934691960021055
R2 :  0.6046198559797608
'''
