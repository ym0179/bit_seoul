#Day8
#2020-11-18

#Fashion-MNIST
'''
10개의 카테고리

0 티셔츠/탑
1 바지
2 풀오버(스웨터의 일종)
3 드레스
4 코트
5 샌들
6 셔츠
7 스니커즈
8 가방
9 앵클 부츠
'''

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (60000, 10)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_fashion.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=32)
print("loss : ", result1[0])
print("acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/fashion-03-0.2569.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################
#2. 모델
model3 = Sequential()
model3.add(Conv2D(64, (3,3), padding="same", input_shape=(28,28,1)))
model3.add(Conv2D(32, (2,2), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weights_fashion.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

'''
save 한 모델:
loss :  0.8111734986305237
acc :  0.9004999995231628
'''
# model1
# loss :  0.8111734986305237
# acc :  0.9004999995231628

# model2
# loss :  0.27602913975715637
# accuracy :  0.9045000076293945

# model3
# loss :  0.8111734986305237
# accuracy :  0.9004999995231628
