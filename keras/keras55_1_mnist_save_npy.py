#Day9
#2020-11-19

#mnist (0-9까지의 손글씨) 예제
# numpy로 데이터 저장하기!!

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

#전처리 이전에 데이터 저장
np.save('./data/mnist_x_train.npy', arr=x_train)
np.save('./data/mnist_x_test.npy', arr=x_test)
np.save('./data/mnist_y_train.npy', arr=y_train)
np.save('./data/mnist_y_test.npy', arr=y_test)

'''
#1. 데이터 전처리 OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
print(y_train[0])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_test02_2.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=32)
print("loss : ", result1[0])
print("acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/11-0.0776.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################
# 2. 모델
model3 = Sequential()
model3.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1)))
model3.add(Conv2D(30, (2,2), padding="same"))
model3.add(Conv2D(20, (2,2), padding="same"))
model3.add(Conv2D(15, (2,2), padding="same"))
model3.add(Conv2D(10, (2,2), padding="same"))
model3.add(Conv2D(5, (2,2), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weight_test02.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

"""
save 한 모델:
loss :  0.09556204080581665
acc :  0.9797999858856201
"""
# model1
# loss :  0.09556204080581665
# acc :  0.9797999858856201

# model2
# loss :  0.06969821453094482
# accuracy :  0.9797000288963318

# model3
# loss :  0.09556204080581665
# accuracy :  0.9797999858856201
'''