#Day28
#2020-12-16

#CAE - CNN AE

import numpy as np
from tensorflow.keras.datasets import mnist, cifar10

(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test/255.

# print(x_train.shape) #(50000, 32, 32, 3)
# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(4,4), padding="valid", input_shape=(32,32,3),
                    activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(4,4), padding="valid", activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(4,4), padding="valid", activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(4,4), padding="valid", activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(4,4), padding="valid", activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(4,4), padding="valid", activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=32*32*3, activation='sigmoid'))
    
    return model

model = autoencoder(hidden_layer_size=512) 

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
# model.compile(optimizer='adam', loss="mse", metrics=['acc'])

model.fit(x_train, x_train.reshape(50000,32*32*3), epochs=5, batch_size=256)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10))= plt.subplots(2,5,figsize=(20,7))


#이미지 다섯개를 무작위로 고름
random_images = random.sample(range(output.shape[0]),5)

#원본(입력) 이미지를 맨위에 그린다
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32,32,3), cmap='gray')
    if i ==0 :
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(32,32,3), cmap='gray')
    if i ==0 :
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()