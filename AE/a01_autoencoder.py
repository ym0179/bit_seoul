#Day28
#2020-12-16

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) #차원축소 == PCA를 딥러닝으로 구현??
decoded = Dense(784, activation='sigmoid')(encoded) #차원축소 했다가 증폭하는 느낌

autoencoder = Model(input_img, decoded)
# autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2) #자기 자신을 확인

decoded_imgs = autoencoder.predict(x_test) #x_test 넣으면 x_test가 나와야함

# 이미지 그려서 확인
import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
