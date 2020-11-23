#Day11
#2020-11-23

'''
선형 서포트벡터 머신 (Linear SVM) - 다중분류문제
- SVM 은 클래스를 구분하는 분류 문제에서, 각 클래스를 잘 구분하는 선을 그어주는 방식
- 원핫인코딩 필요없음
- 두 클래스의 가운데 선을 그어주게 되고, 가장 가까이 있는 점들과의 거리가 가장 큰 직선을 찾음
- 이때 가장 가까이 있는 점들을 Support Vector 라고 하고, 찾은 직선과 서포트벡터 사이의 거리를 최대 마진(margin) 이라 함
- 마진을 최대로 하는 서포트벡터와 직선을 찾는 것이 목표
'''

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC

#####1. 데이터
dataset = load_iris()
x, y = load_iris(return_X_y=True) #자동으로 x,y 넣어줌
# x = dataset.data
# y = dataset.target

# #OneHotEncoding
# y = to_categorical(y)

#train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=44)
# x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=44)

#scaling
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


####2. 모델링
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# model = Sequential()
# model.add(Dense(64, activation='relu',input_shape=(4,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='softmax'))
model = LinearSVC()


#####3. 컴파일, 훈련
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
# es = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
# model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,validation_data=(x_val,y_val)) 
model.fit(x_train,y_train)


#####4. 평가
# loss,acc = model.evaluate(x_test,y_test,batch_size=1)
# print("loss : ",loss)
# print("acc : ",acc)
result = model.score(x_test,y_test)
print("score :" , result)


#####5. 예측
predict = model.predict(x_test)


