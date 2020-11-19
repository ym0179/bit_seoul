#Day9
#2020-11-19

#다중분류
import numpy as np
# from sklearn.datasets import load_iris
# dataset = load_iris()
# x = dataset.data
# y = dataset.target
# # print(x)
# # print(x.shape, y.shape) #(150, 4) (150,)


#저장한 데이터 불러오기
x=np.load('./data/iris_x.npy')
y=np.load('./data/iris_y.npy')


#1. 전처리

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state = 44)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#reshape
x_train = x_train.reshape(x_train.shape[0], 4, 1, 1)
x_val = x_val.reshape(x_val.shape[0], 4, 1, 1)
x_test = x_test.reshape(x_test.shape[0], 4, 1, 1)


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_iris.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result1[0])
print("mae : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/iris-20-0.0159.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result2[0])
print("mae : ", result2[1])


################ 3. load_weights ##################
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model3 = Sequential()
model3.add(Conv2D(32, (1,1), padding="same", input_shape=(4,1,1)))
model3.add(Conv2D(16, (1,1), padding="same"))
model3.add(MaxPooling2D(pool_size=1))
model3.add(Conv2D(8, (1,1), padding="same"))
model3.add(MaxPooling2D(pool_size=1))
model3.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model3.load_weights('./save/weights_iris.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result3[0])
print("mae : ", result3[1])

'''
save 한 모델:
loss :  0.2871306538581848
acc :  0.9333333373069763
'''
# model1
# loss :  0.2871306538581848
# mae :  0.9333333373069763

# model2
# loss :  0.05457454174757004
# mae :  0.9666666388511658

# model3
# loss :  0.2871306538581848
# mae :  0.9333333373069763
