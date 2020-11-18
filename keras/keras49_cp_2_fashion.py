#Day8
#2020-11-18

#Fashion-MNIST
'''
10개의 카테고리

0 티셔츠/탑
1 바지
2 풀오버(스웨터의 일종)
3 드레스
4 코트
5 샌들
6 셔츠
7 스니커즈
8 가방
9 앵클 부츠
'''

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (60000, 10)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#predict 만들기
x_pred = x_test[:10]
y_pred = y_test[:10]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
model = Sequential()
model.add(Conv2D(64, (3,3), padding="same", input_shape=(28,28,1)))
model.add(Conv2D(32, (2,2), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#다중분류에서는 loss가 categorical crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/fashion-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(patience=20,mode='auto',monitor='val_loss')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es,cp])

model.save('./save/model_fashion.h5')
model.save_weights('./save/weights_fashion.h5')

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
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
loss :  0.8111734986305237
acc :  0.9004999995231628
예측값 :  [9 2 1 1 6 1 4 6 5 7]
실제값 :  [9 2 1 1 6 1 4 6 5 7]
'''