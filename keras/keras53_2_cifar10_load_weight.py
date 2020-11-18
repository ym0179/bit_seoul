#Day8
#2020-11-18

#CIFAR-10 dataset은 32x32픽셀의 60000개 컬러이미지가 포함되어있으며, 각 이미지는 10개의 클래스로 라벨링

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar10.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result1[0])
print("acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cifar10-03-0.7985.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################
#2. 모델
model3 = Sequential()
model3.add(Conv2D(32, (3,3), padding="same", input_shape=(32,32,3)))
model3.add(Conv2D(32, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Conv2D(64, (3,3), padding="same"))
model3.add(Conv2D(64, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Conv2D(128, (3,3), padding="same"))
model3.add(Conv2D(128, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(256, activation='relu'))
model3.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weights_cifar10.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

'''
save 한 모델:
loss :  2.2751755714416504
acc :  0.6973000168800354
'''
# model1
# loss :  2.2751755714416504
# acc :  0.6973000168800354

# model2
# loss :  0.8113862872123718
# accuracy :  0.7233999967575073

# model3
# loss :  2.2751755714416504
# accuracy :  0.6973000168800354