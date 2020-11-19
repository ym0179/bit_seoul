#Day9
#2020-11-19

#이진분류: 암 예측 0또는 1
#이진분류는 sigmoid, one hot encoding 필요없음, binary_cross_entropy (loss)
import numpy as np
# from sklearn.datasets import load_breast_cancer
# dataset = load_breast_cancer()
# x = dataset.data
# y = dataset.target
# print(x)
# print(x.shape, y.shape) #(569, 30) (569,)

#저장한 데이터 불러오기
x=np.load('./data/cancer_x.npy')
y=np.load('./data/cancer_y.npy')


#1. 전처리

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state = 44)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cancer.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cancer-04-0.0349.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model3 = Sequential()
model3.add(Dense(64, activation='relu',input_shape=(30,)))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(16, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model3.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weights_cancer.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

'''
save 한 모델:
loss :  0.07271450757980347
acc :  0.9824561476707458
'''
# model1
# loss :  0.07271450757980347
# accuracy :  0.9824561476707458

# model2
# loss :  0.03869706392288208
# accuracy :  0.9912280440330505

# model3
# loss :  0.07271450757980347
# accuracy :  0.9824561476707458
