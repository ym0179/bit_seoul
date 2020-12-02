#Day18
#2020-11-18

#mnist (0-9까지의 손글씨) 예제
#OneHotEncoding

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리 OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #마지막은 채널 1 (흑백)
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2
model = Sequential()
model.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1))) #output: (28,28,10)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(30, (2,2), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(20, (2,2), padding="same", kernel_regularizer=l1(0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(15, (2,2), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(patience=3,mode='auto',monitor='loss')
to_hist = TensorBoard(
    log_dir= "graph",
    histogram_freq=0,
    write_graph=True,
    write_images=True
)
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es,to_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

'''
1) weight regularization(가중치 규제) : 가중치가 클수록 큰 패널티를 부과하여 Overfitting을 억제하는 방법
kernel_regularization / bias_regularization :  기본값은 규제를 적용하지 않는 것

    - L1 = 절대값 규제
    대부분의 요소값이 0인 sparse feature에 의존한 모델에서 불필요한 feature에 대응하는 가중치를 0으로 만들어 해당 feature를 모델이 무시하게 만듬
    - L2 = 제곱값 규제
    아주 큰 값이나 작은 값을 가지는 outlier모델 가중치에 대해 0에 가깝지만 0은 아닌값으로 만듬
    선형모델의 일반화능력을 개선시키는 효과
    -> kernel_regularizer=regularizers.l2(0.001) : 가중치 행렬의 모든 원소를 제곱하고 0.001을 곱하여 네트워크의 전체 손실에 더해진다는 의미, 이 규제(패널티)는 훈련할 때만 추가 / 0.001은 규제의 양을 조절

- 패널티에 대한 효과를 크게보기 위해 L1보다 L2를 많이 사용하는 경향이 있음
- dropout도 regularization 효과

3) weight initialization(가중치 초기화) : 기본값은 "glorot_uniform" 초기화
kernel_initializer
bias_initializer

4) BatchNormalization: 배치 정규화(BN)은 초기값이 아니라 출력값을 정규화하여 원활한 학습과 기울기 소실의 문제를 해결하고자 하는 것
- activation 쓰기 전에 쓰는게 좋음
- Dropout과 같이 쓸 필요가 없음
- Use Large Learning Rates

5) Dropout(n) : n은 dropout 비율
- 0.2-0.5가 적당
- 큰 network가 좋다

https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/

<추가 자료>
1.initializer = 초기화
연산을 시작할때 가중치 또는 bias 초기값을 설정

2.regularizer = 정규화
연산하면서 가중치같은걸 layer 마다 업데이트 하는데 
그때마다 나의 목표에 맞춰서 연산할 가중치 또는 bias에 제한을 줌 
ex) 가중치가 이상해질거같을때 제한을 주어 이상해지지 않게해줌

3.BatchNomalization = 일반화(정규화라고 불릴때도 있음)
한 layer에서 다음 layer로 넘어갈때 노드에서 연산된 값을 정규화 시켜줌
0~1 사이의 값으로 scale 기능도 있고 parameter값에 영향을 크게 안받아서
learning rate값을 높게 해도 잘 받아줌 -> 학습속도가 빨라짐
'''