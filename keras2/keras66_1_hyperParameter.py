#Day18
#2020-11-18

# GridSearchCV랑 kears 모델 엮어보기

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
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

x_train = x_train.reshape(60000, 28*28).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam',learn_rate=0.01, momentum=0): #batch_size는 선언 안해도 되는것?
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],
                loss="categorical_crossentropy") #onehotencoding 안해주면 sparse_categorical_crossentropy
    return model

def create_hyperparameters():
    batches = [1,8,16,32,64]
    optimizers = ['rmsprop','adam','adadelta','SGD']
    dropout = np.linspace(0.1,0.5,5).tolist()
    # dropout = [0.1,0.2,0.3,0.4,0.5]
    epochs = [100, 200, 500]
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3] #loss 최적화 시키는 optimizer에 lr
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'] #weight initializer
    # return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, 'epochs' : epochs}
    return dict(learn_rate=learn_rate, momentum=momentum, batch_size=batches, 
                optimizer=optimizers, drop=dropout, epochs=epochs, init_mode=init_mode)

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=1)

# search = GridSearchCV(model, hyperparameters, cv=2)
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train,y_train) 
# TypeError: estimator should be an estimator implementing 'fit' method, <function build_model at 0x0000020D6504F940> was passed
# keras 모델을 GridSearchCV(sklearn)에 넣어서 fit 하니까 인식을 못함 -> wrappers.scikit_learn 통해서 해결

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)

'''
GridSearchCV
{'batch_size': 30, 'drop': 0.1, 'optimizer': 'adam'}
최종 스코어 :  0.9675999879837036


RandomizedSearchCV
'''

# https://stackoverflow.com/questions/59746974/cannot-clone-object-tensorflow-python-keras-wrappers-scikit-learn-kerasclassifi