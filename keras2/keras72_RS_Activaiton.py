#Day18
#2020-11-18

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
from tensorflow.keras.layers import Activation, LeakyReLU, ELU, ReLU, ThresholdedReLU
from tensorflow.keras.activations import relu, selu, elu, tanh, sigmoid, linear, softplus, softsign

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 
# OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer=Adam ,learn_rate=0.01, activation=relu):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, name='hidden2')(x) #linear default여서 똑같은 값 던져줌
    x = Activation(activation)(x)
    x = Dropout(drop)(x)
    x = Dense(128, name='hidden3')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr = learn_rate),
                  metrics=['acc'],
                  loss="categorical_crossentropy")
    # print(model.summary())
    return model

def create_hyperparameters():
    batches = [8,16,32,64]
    optimizers = [Adam, RMSprop, SGD]
    dropout = np.linspace(0.1,0.5,5).tolist()
    epochs = [100,300,500]
    learn_rate = [0.01,0.001,0.0001]
    # activation = [LeakyReLU, ELU, ReLU, ThresholdedReLU]
    activation = [relu, selu, elu, tanh, sigmoid, linear, softplus, softsign]
    return dict(learn_rate=learn_rate, batch_size=batches, activation=activation,
                optimizer=optimizers, drop=dropout, epochs=epochs)

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=1)

# search = GridSearchCV(model, hyperparameters, cv=2)
search = RandomizedSearchCV(model, hyperparameters, cv=2)
search.fit(x_train,y_train) 

acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)

# summarize results
print("Best: %f using %s" % (search.best_score_, search.best_params_))
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
params = search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
