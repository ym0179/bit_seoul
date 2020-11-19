#Day9
#2020-11-19

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(569, 30) (569,)

#1. 전처리

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
x_train = x_train.reshape(x_train.shape[0], 30, 1)
x_val = x_val.reshape(x_val.shape[0], 30, 1)
x_test = x_test.reshape(x_test.shape[0], 30, 1)

#predict 만들기 - train-test 에서 shuffle 하고 나서 해줌
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델링
model = Sequential()
model.add(Conv1D(64, (2), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(32, (2), padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
modelpath = './model/cancer_conv1d.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,
        callbacks=[es,cp],validation_data=(x_val,y_val)) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/cancer_conv1d.hdf5')

#4. 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

#5. 예측
result = model.predict(x_pred)
print("예측값 : ", result.T.reshape(10,))
print("실제값 : ", y_pred)


'''
LSTM 모델
loss :  0.3498838245868683
acc :  0.9210526347160339
예측값 :  [0.99999964 0.9999999  0.00171117 0.9987224  0.99979585 0.01519871
 0.0037073  0.04073161 0.99980134 0.00277037]
실제값 :  [1 1 0 1 1 0 0 0 1 0]

Conv1D 모델
loss :  0.04970039427280426
acc :  0.9824561476707458
예측값 :  [9.9985206e-01 9.9985456e-01 3.2701082e-05 9.7736520e-01 9.9150276e-01
 9.8298073e-01 1.1511337e-05 9.9999988e-01 9.9999201e-01 3.0906026e-08]
실제값 :  [1 1 0 1 1 1 0 1 1 0]
'''

