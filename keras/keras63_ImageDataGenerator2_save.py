#Day14
#2020-11-26

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

np.random.seed(33) #numpy random 값 쓰는 경우 다 33에 있는 난수 씀

# 이미지 생성 옵션 정하기
# 이미지 augmentation
# 학습 이미지에 적용한 augmentation 인자를 지정
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip=True, #50% 확률로 수평으로 뒤집음
                                   vertical_flip=True, #50% 확률로 수직으로 뒤집음
                                   width_shift_range=0.1, #왼쪽, 오른쪽 움직임 (평행 이동)
                                   height_shift_range=0.1, #위, 아래 움직임 (평행 이동)
                                   rotation_range=5, #n도 안에서 랜덤으로 이미지 회전
                                   zoom_range=1.2, #range 안에서 랜덤하게 zoom
                                   shear_range=0.7, #0.7 라디안 내외로 시계반대방향으로 변형
                                   fill_mode='nearest', #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
                                   )

# 검증 및 테스트 이미지는 augmentation을 적용하지 않음
# 모델 성능을 평가할 때에는 이미지 원본을 사용
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업
# 이미지를 배치 단위로 불러와 줄 generator
xy_train = train_datagen.flow_from_directory(
    './data/data1/train', # target directory
    target_size=(150,150), # 모든 이미지의 크기가 150x150로 조정
    batch_size=2,
    class_mode='binary' # 이진분류 / categorical 다중분류
    # binary_crossentropy 손실 함수를 사용하므로 binary 형태로 라벨
    # , save_to_dir='./data/data1_2/train'
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=2,
    class_mode='binary' #이진분류 / categorical 다중분류
)

# print("==========================================")
# print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(xy_train[0])
# # print(xy_train[0].shape) #error
# print(xy_train[0][0]) #x값
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(xy_train[0][0].shape) #(5, 150, 150, 3) -> 5는 batch size
# print(xy_train[0][1]) #y값
# print(xy_train[0][1].shape) #(5,) -> 5는 batch size
# # print(xy_train[1][0].shape) #(5, 150, 150, 3)
# # print(xy_train[1][1].shape) #(5,)
# print(len(xy_train)) #32 * 5 = 총 160개 train 이미지

# print("==========================================")
# print(xy_train[0][0][0])

# batch_size 풀로 잡고 [0][0]으로 numpy에 저장
# numpy로 저장하면 시간 절약됨
# np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
# np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
# np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
# np.save('./data/keras63_test_y.npy', arr=xy_test[0][1])

model = Sequential()
model.add(Conv2D(32, (1,1), padding="same", input_shape=(150,150,3)))
model.add(Conv2D(16, (1,1), padding="same"))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(8, (1,1), padding="same"))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

# 학습
hist = model.fit_generator(
    xy_train,
    steps_per_epoch=50, #augmentation한거에서 100개만 뽑음 /= 제너레이터로부터 얼마나 많은 샘플을 뽑을 것인지
    epochs=100,
    validation_data=xy_test,
    validation_steps=4,
    verbose=1
)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']


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

