#Day14
#2020-11-26

#남자 여자 데이터
#넘파이 저장
#fit_generator로 코딩

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

np.random.seed(33)

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip=True, #50% 확률로 수평으로 뒤집음
                                   vertical_flip=True, #50% 확률로 수직으로 뒤집음
                                   width_shift_range=0.1, #왼쪽, 오른쪽 움직임 (평행 이동)
                                   height_shift_range=0.1, #위, 아래 움직임 (평행 이동)
                                   rotation_range=5, #n도 안에서 랜덤으로 이미지 회전
                                   zoom_range=1.2, #range 안에서 랜덤하게 zoom
                                   shear_range=0.7, #0.7 라디안 내외로 시계반대방향으로 변형
                                   fill_mode='nearest', #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
                                   validation_split=0.2
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './data/data2', # target directory
    target_size=(200,200), 
    batch_size=8,
    class_mode='binary', # 이진분류
    subset='training'
)

xy_valid = train_datagen.flow_from_directory(
    './data/data2', # target directory
    target_size=(200,200),
    batch_size=8,
    class_mode='binary', # 이진분류 
    subset='validation'
)
# test valid 나누기 전
# print(xy_train[0][0].shape) #(1736, 300, 300, 3)

# test valid 나누기 후
# print(xy_train[0][0].shape) #(1389, 300, 300, 3)
# print(xy_valid[0][0].shape) #(347, 300, 300, 3)

# numpy 저장
# np.save('./data/keras64_x.npy', arr=xy_train[0][0])
# np.save('./data/keras64_y.npy', arr=xy_train[0][1])

model = Sequential()
model.add(Conv2D(64, (3,3), padding="same", input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(16, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

# 학습
hist = model.fit_generator(
    xy_train,
    steps_per_epoch=100, #augmentation한거에서 100개만 뽑음 /= 제너레이터로부터 얼마나 많은 샘플을 뽑을 것인지
    #보통은 데이터셋의 샘플 수를 배치 크기로 나눈 값
    epochs=50,
    validation_data=xy_valid,
    validation_steps=50, #한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정
    #보통은 검증 데이터셋의 샘플 수를 배치 크기로 나눈 값
    verbose=2
)

# scores = model.evaluate_generator(test_generator, steps=5)
# output = model.predict_generator(test_generator, steps=5)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']

# print(acc[-1])
# print(val_acc[-1])


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
plt.legend(['acc','val_acc']) #location 명시 안하면 알아서 빈자리에 박스 그림

plt.show()
