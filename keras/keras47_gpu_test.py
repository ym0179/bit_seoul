#Day8
#2020-11-18

import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import datetime

num_samples = 100
height = 71
width = 71
num_classes = 100

start1 = datetime.datetime.now()
with tf.device('/gpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3), classes=num_classes)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    # Generate dummy data
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    model.fit(x, y, epochs=3, batch_size=15)

    model.save('my_model.h5')

end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('GPU 처리시간 : ', time_delta1) #GPU 처리시간 :  0:00:11.697386

start2 = datetime.datetime.now()
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3), classes=num_classes)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    # Generate dummy data
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    model.fit(x, y, epochs=3, batch_size=15)

    model.save('my_model.h5')

end2 = datetime.datetime.now()
time_delta2 = end2 - start2
print('CPU 처리시간 : ', time_delta2) #CPU 처리시간 :  0:00:20.126813

