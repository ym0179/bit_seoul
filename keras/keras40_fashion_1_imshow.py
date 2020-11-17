#Day7
#2020-11-17

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
print(x_train[-1])
print("y_train[-1] : ", y_train[-1])

plt.imshow(x_train[-1], 'gray') #이미지 확인
plt.show()

