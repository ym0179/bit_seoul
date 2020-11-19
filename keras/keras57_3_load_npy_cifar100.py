#Day9
#2020-11-19

#just like the CIFAR-10, except it has 100 classes containing 600 images each 
#There are 500 training images and 100 testing images per class
#The 100 classes in the CIFAR-100 are grouped into 20 superclasses
#Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, Dropout

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

#저장한 데이터 불러오기
x_train=np.load('./data/cifar100_x_train.npy')
x_test=np.load('./data/cifar100_x_test.npy')
y_train=np.load('./data/cifar100_y_train.npy')
y_test=np.load('./data/cifar100_y_test.npy')

#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


#2. 모델링
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar100.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result1[0])
print("acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cifar100-131-2.2189.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################
#2. 모델
model3 = Sequential()
model3.add(Conv2D(64, (3,3), padding="same", input_shape=(32,32,3)))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.3))
model3.add(Conv2D(64, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.3))
model3.add(Conv2D(128, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.4))
model3.add(Conv2D(256, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.5))
model3.add(Conv2D(64, (3,3), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weights_cifar100.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=64)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

'''
save 한 모델:
loss :  2.3746769428253174
acc :  0.38260000944137573
'''
# model1
# loss :  2.3746769428253174
# acc :  0.38260000944137573

# model2
# loss :  2.181558847427368
# accuracy :  0.4207000136375427

# model3
# loss :  2.3746769428253174
# accuracy :  0.38260000944137573

