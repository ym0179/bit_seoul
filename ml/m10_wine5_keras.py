#Day11
#2020-11-23

#winequality-white.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder


# 1. 데이터
#pandas로 csv 불러오기
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')

#x,y 값 나누기
y = wine['quality']
x = wine.drop('quality',axis=1)
# print(x)
# print(y)
# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

#인위적으로 y의 labeling 값을 조절 (분포를 더 작게 잡아주기)
#quality 0-4까지는 0, 5-7은 1, 8-9는 2
newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

print(newlist)
y = newlist

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

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
model.add(Dense(3, activation='softmax'))

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
loss :  0.33066290616989136
acc :  0.9214285612106323
예측값 :  [1 1 1 1 1 1 1 1 1 1]
실제값 :  [1 1 1 1 1 1 1 1 1 1]
'''