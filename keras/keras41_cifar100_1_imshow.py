#Day7
#2020-11-17

#just like the CIFAR-10, except it has 100 classes containing 600 images each 
#There are 500 training images and 100 testing images per class
#The 100 classes in the CIFAR-100 are grouped into 20 superclasses
#Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)
print(x_train[-1])
print("y_train[-1] : ", y_train[-1])

plt.imshow(x_train[-1], 'gray') #이미지 확인
plt.show()

