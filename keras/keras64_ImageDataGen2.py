#Day14
#2020-11-26

#넘파이 불러와서
#.fit으로 코딩

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

#저장한 데이터 불러오기
x=np.load('./data/keras64_x.npy')
y=np.load('./data/keras64_y.npy')
print(x.shape) #(1736, 200, 200, 3)
print(y.shape) #(1736,)


#1. 전처리
#train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=77)

#scaling은 ImageDataGenerator 사용할 때 이미 해줌

#predict 만들기 - train-test 에서 shuffle 하고 나서 해줌
x_pred = x_test[:20]
y_pred = y_test[:20]


#2. 모델링
model = Sequential()
model.add(Conv2D(128, (3,3), padding="same", input_shape=(200,200,3), activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,2), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,2), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(63, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='val_loss',patience=100,mode='auto')
modelpath = './model/keras64.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=32,verbose=2,callbacks=[es,cp],validation_data=(x_val,y_val)) 

# 모델 불러오기
model = load_model('./model/keras64.hdf5')

#4. 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

#5. 예측
result = model.predict(x_pred)

# print("예측값 : ", result.T.reshape(10,))
# print("실제값 : ", y_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행

y_predicted = list(map(int, y_predicted)) #보기 쉽게
y_pred = list(map(int, y_pred)) #보기 쉽게

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred)

'''
loss :  0.8301668763160706
acc :  0.6321839094161987
예측값 :  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
실제값 :  [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]

loss :  0.6671748757362366
acc :  0.6264367699623108
예측값 :  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
실제값 :  [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''


