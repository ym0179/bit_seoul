#Day8
#2020-11-18

#just like the CIFAR-10, except it has 100 classes containing 600 images each 
#There are 500 training images and 100 testing images per class
#The 100 classes in the CIFAR-100 are grouped into 20 superclasses
#Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

#predict 만들기
x_pred = x_test[:20]
y_pred = y_test[:20]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(64, (3,3), padding="same", input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(256, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), padding="same"))
# model.add(Conv2D(64, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
# model.add(Conv2D(32, (3,3), padding="same"))
# model.add(Conv2D(32, (3,3), padding="same"))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#다중분류에서는 loss가 categorical crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/cifar100-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(patience=30,mode='auto',monitor='val_loss')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=2, validation_split=0.2, callbacks=[es,cp])

model.save('./save/model_cifar100.h5')
model.save_weights('./save/weights_cifar100.h5')


#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

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

result = model.predict(x_pred)

y_predicted = np.argmax(result,axis=1) # axis가 0 이면 열, axis가 1이면 행
y_pred_recovery = np.argmax(y_pred, axis=1) #원핫인코딩 원복

print("예측값 : ", y_predicted)
print("실제값 : ", y_pred_recovery)

'''
loss :  2.3746769428253174
acc :  0.38260000944137573
예측값 :  [95 33 95 66 71  7 63  7 60 83 87 27 81 69 40 63 92 63 70 53]
실제값 :  [49 33 72 51 71 92 15 14 23  0 71 75 81 69 40 43 92 97 70 53]
'''