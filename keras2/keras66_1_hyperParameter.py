#Day18
#2020-11-18

# GridSearchCV랑 kears 모델 엮어보기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization, Activation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.regularizers import l1, l2, l1_l2


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 
# OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
# print(y_train[0])

x_train = x_train.reshape(60000, 28*28).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, 
                optimizer=Adam,
                learn_rate=0.01, 
                activation='relu',
                init_mode='uniform',
                neurons=64,
                regularization_fn=l1,
                layer_num=1,
                stop=None) : #batch_size는 선언 안해도 되는것? -> 원래 default값 있음 32

    inputs = Input(shape=(28*28,))
    
    for i in range(layer_num):
        if i == 0:
            x = Dense(neurons, kernel_initializer=init_mode, 
                    kernel_regularizer=regularization_fn(0.01))(inputs)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
        else:
            x = Dense(neurons, kernel_initializer=init_mode, 
        kernel_regularizer=regularization_fn(0.01))(x)
            x = Activation(activation)(x)
            x = Dropout(drop)(x)

    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr = learn_rate),
                  metrics=['acc'],
                  loss="categorical_crossentropy") #onehotencoding 안해주면 sparse_categorical_crossentropy
    print(model.summary())
    return model

def create_hyperparameters():
    layer_num = [1,2]
    neurons = [32, 64, 128, 256]
    batches = [16,32]
    optimizers = [Adam, Adadelta, RMSprop, SGD, Nadam]
    dropout = [0.2,0.3,0.4,0.5]
    epochs = [100,300]
    learn_rate = [0.1, 0.01,0.001] #loss 최적화 시키는 optimizer에 lr
    activation = ['relu', 'selu','elu','tanh']
    regularization_fn = [l1,l2,l1_l2]
    init_mode = ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'] #weight initializer
    return dict(learn_rate=learn_rate, batch_size=batches, activation=activation,
                optimizer=optimizers, drop=dropout, epochs=epochs, neurons=neurons, regularization_fn=regularization_fn, init_mode=init_mode, layer_num=layer_num)


hyperparameters = create_hyperparameters()

stopper = EarlyStopping(monitor='val_acc', patience=3)
fit_params = dict(callbacks=[stopper])

model = KerasClassifier(build_fn=build_model, verbose=1)
# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train,y_train, **fit_params) 
# TypeError: estimator should be an estimator implementing 'fit' method, <function build_model at 0x0000020D6504F940> was passed
# keras 모델을 GridSearchCV(sklearn)에 넣어서 fit 하니까 인식을 못함 -> wrappers.scikit_learn 통해서 해결

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)


# https://stackoverflow.com/questions/59746974/cannot-clone-object-tensorflow-python-keras-wrappers-scikit-learn-kerasclassifi