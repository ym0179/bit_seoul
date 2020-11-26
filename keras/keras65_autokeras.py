#Day14
#2020-11-26

# pip install autokeras
# pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

x_train = np.load("./data/keras63_train_x.npy")
x_test = np.load("./data/keras63_test_x.npy")
y_train = np.load("./data/keras63_train_y.npy")
y_test = np.load("./data/keras63_test_y.npy")

print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

# Initialize the image classifier
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=3
)
# Feed the image classifier with training data
clf.fit(x_train, y_train, epochs=50)

# Predict with the best model
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data
print(clf.evaluate(x_test, y_test))
# clf.summary()