#Day18
#2020-12-02

# GridSearchCV랑 kears 모델 엮어보기

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, LSTM
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 
# OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)
# print(y_train[0])

x_train = x_train.astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam', learn_rate=0.01, activation='relu'): 
    inputs = Input(shape=(28,28), name='input')
    x = LSTM(128, activation='relu', return_sequences=True, name='hidden1')(inputs)
    x = LSTM(64, activation='relu', name='hidden2')(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu', name='hidden5')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],
                loss="categorical_crossentropy") #onehotencoding 안해주면 sparse_categorical_crossentropy
    return model

def create_hyperparameters():
    batches = [1,8,16,32,64]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = np.linspace(0.1,0.5,5).tolist()
    epochs = [100, 200, 500]
    learn_rate = [0.1, 0.01,0.001]
    activation = ['relu', 'selu','elu','tanh']
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, 'epochs' : epochs, 'learn_rate' : learn_rate, 'activation':activation}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=1)

search = GridSearchCV(model, hyperparameters, cv=2)
# search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train,y_train) 
# TypeError: estimator should be an estimator implementing 'fit' method, <function build_model at 0x0000020D6504F940> was passed
# keras 모델을 GridSearchCV(sklearn)에 넣어서 fit 하니까 인식을 못함 -> wrappers.scikit_learn 통해서 해결

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)


# https://stackoverflow.com/questions/59746974/cannot-clone-object-tensorflow-python-keras-wrappers-scikit-learn-kerasclassifi