#Day8
#2020-11-18

#이진분류: 암 예측 0또는 1
#이진분류는 sigmoid, one hot encoding 필요없음, binary_cross_entropy (loss)
import numpy as np
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(569, 30) (569,)

#1. 전처리

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

#predict 만들기 - train-test 에서 shuffle 하고 나서 해줌
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(30,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/cancer-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es,cp],validation_data=(x_val,y_val)) 

model.save('./save/model_cancer.h5')
model.save_weights('./save/weights_cancer.h5')


#4. 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) #인치 단위
#1번째 그림
plt.subplot(2, 1, 1) #2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #plt.plot에서 명시한 label이 박스형태로 상단 오른쪽에 나옴

#2번째 그림
plt.subplot(2, 1, 2) #2행 1열 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) #location명시 안하면 알아서 빈자리에 박스 그림

plt.show()

#5. 예측
result = model.predict(x_pred)

print("예측값 : ", result.T.reshape(10,))
print("실제값 : ", y_pred)
'''
loss :  0.07271450757980347
acc :  0.9824561476707458
예측값 :  [9.0638277e-05 4.1658999e-17 9.9989963e-01 5.7662373e-19 9.9703872e-01
 9.9929082e-01 1.0000000e+00 9.9956053e-01 7.8910510e-11 1.0000000e+00]
실제값 :  [0 0 1 0 1 1 1 1 0 1]
'''