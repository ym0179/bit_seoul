#Day14
#2020-11-26

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=5,
    class_mode='binary' # 이진분류 / categorical 다중분류
    # binary_crossentropy 손실 함수를 사용하므로 binary 형태로 라벨
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary' #이진분류 / categorical 다중분류
)

# 학습
model.fit_generator(
    xy_train,
    steps_per_epoch=100, #augmentation한거에서 100개만 뽑음
    epochs=20,
    validation_data=xy_test,
    validation_steps=4
)