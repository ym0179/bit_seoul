#Day9
#2020-11-19

#당뇨병 진행도 예측 모델 학습 (sklearn 라이브러리)
'''
[x]
442 행 10 열 
Age
Sex
Body mass index
Average blood pressure
S1
S2
S3
S4
S5
S6

[y]
442 행 1 열 
target: a quantitative measure of disease progression one year after baseline
'''

# from sklearn.datasets import load_diabetes
# dataset = load_diabetes()
# x = dataset.data
# y = dataset.target
# print(x)
# print(x.shape, y.shape) #(442, 10) (442,)
# # print(dataset)

import numpy as np 

#저장한 데이터 불러오기
x=np.load('./data/diabetes_x.npy')
y=np.load('./data/diabetes_y.npy')


#1. 전처리
#train-test split -> scaling train 만 하니까 젤 먼저 split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state = 44)

#scaling - 2차원 input만 가능 -> reshape 나중에
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_diabetes.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result1[0])
print("mae : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/diabetes-24-2153.2610.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result2[0])
print("mae : ", result2[1])


################ 3. load_weights ##################
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model3 = Sequential()
model3.add(Dense(64, activation='relu',input_shape=(10,)))
model3.add(Dropout(0.3))
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(1))

#3. 컴파일, 훈련
model3.compile(loss="mse", optimizer="adam", metrics=["mae"])
model3.load_weights('./save/weights_diabetes.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result3[0])
print("mae : ", result3[1])

'''
save 한 모델:
loss :  1894.115234375
mae :  33.49161148071289
'''
# model1
# loss :  1894.115234375
# mae :  33.49161148071289

# model2
# loss :  2201.29443359375
# mae :  36.161949157714844

# model3
# loss :  1894.115234375
# mae :  33.49161148071289

