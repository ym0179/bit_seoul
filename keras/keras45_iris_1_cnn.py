#Day8
#2020-11-18

#다중분류
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

#1. 전처리

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7)

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

# print(x_train.shape) #(120, 4, 1, 1)
# print(x_test.shape) #(30, 4, 1, 1)

#predict 만들기
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(32, (1,1), padding="same", input_shape=(4,1,1)))
model.add(Conv2D(16, (1,1), padding="same"))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(8, (1,1), padding="same"))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=15,mode='auto')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 


#4. 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

#5. 예측
result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
loss :  0.06635227054357529
acc :  0.9666666388511658
예측값 :  [0 2 2 1 0 2 2 0 1 1]
실제값 :  [0 2 2 1 0 2 2 0 1 1]
'''


