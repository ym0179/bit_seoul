#Day11
#2020-11-23

#winequality-white.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder


# 1. 데이터
#pandas로 csv 불러오기
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
# print(datasets)
# print(datasets.shape) #(4898, 12)
print(datasets['quality'].value_counts())

# pandas dataframe를 numpy 배열로 변환하기
datasets = datasets.to_numpy()

# x,y 데이터 나누기
x = datasets[:,:-1]
y = datasets[:,-1]
# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

# print(y)

#OneHotEncoding
# 1. TensorFlow의 keras.utils.to_categorical()을 이용한 one-hot-encoding
# from tensorflow.keras.utils import to_categorical #index가 0부터 시작함 -> category 10개로 인식 (0부터 9까지)
# y = to_categorical(y)

# 2. Scikit-Learn의 OneHotEncoder 이용한 one-hot-encoding
# 2차원 데이터로 변환
'''
총 n개의 원소가 들어있는 배열 x에 대해서 x.reshape(-1, 정수) 를 해주면 
'열(column)' 차원의 '정수'에 따라서 n개의 원소가 빠짐없이 배치될 수 있도록 
'-1'이 들어가 있는 '행(row)' 의 개수가 가변적으로 정해짐
'''
print(y.shape) #(4898,)
# y = y.reshape(-1,2) 
# print(y.shape) #(2449, 2)
y = y.reshape(-1,1) 
print(y.shape) #(4898, 1)

ohe = OneHotEncoder(sparse=False).fit(y)
y = ohe.transform(y)

# 3. pandas의 get_dummies()를 이용한 one-hot-encoding => df 인 상태에서 사용 (numpy 배열로 변환전이나 df로 전처리 할 때? 아님 사용 후 배열로 바꾸면 될듯)
# y = pd.get_dummies(y)

# print(y)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=66, train_size=0.8)

# scale
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 2. 모델 
model = Sequential()
model.add(Dense(128, activation='relu',input_shape=(11,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 
model.fit(x_train,y_train)

# 4. 예측, 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_test[:10])
y_predict = np.argmax(y_predict,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_test_recovery = np.argmax(y_test[:10], axis=1) #원핫인코딩 원복

print("예측값 : ", y_predict)
print("실제값 : ", y_test_recovery)

'''
loss :  1.3669817447662354
acc :  0.6102041006088257
예측값 :  [7 7 6 6 6 5 5 7 6 6]
실제값 :  [6 6 6 6 6 5 5 7 5 6]
'''