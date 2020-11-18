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


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/iris-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2,callbacks=[es,cp],validation_data=(x_val,y_val)) 

model.save('./save/model_iris.h5')
model.save_weights('./save/weights_iris.h5')

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

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
acc: 0.9667
loss :  0.09106765687465668
acc :  0.9666666388511658
예측값 :  [1 1 2 2 0 0 0 1 1 0]
실제값 :  [1 1 2 2 0 0 0 1 1 0]
'''
