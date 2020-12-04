#Day20
#2020-12-04

#전이학습(transfer)
#: 남이 잘 만든 모델과 가중치를 빼서 쓰겠다
#: 훈련은 안 시키켔다는 뜻 아닌가?

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# include_top:whether to include the 3 fully-connected layers at the top of the network
# This will load the whole VGG16 network, including the top Dense layers.
# Note: by specifying the shape of top layers, input tensor shape is forced
# to be (224, 224, 3), therefore you can use it only on 224x224 images.
# vgg_model = VGG16(weights='imagenet', include_top=True)
# model = VGG16()
# model = VGG16(include_top=False, input_shape=(100, 100, 1) #매개변수 확인해 보기 #138,357,544 parameters
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) #그럼 이렇게 줬을 때랑 파라미터 수가 똑같으면 기본값이 imagenet인가 
                                  #파라미터의 개수가 똑같다고 해서 같은 가중치일 리는 x 연산의 개수만 같을 뿐 
                                  #이미지넷인지 아닌지 어떻게 알아요
                                  #근데 이대로 갖다 쓰면 input layer가... 안 맞을... 텐데? -> input_shape 바꿔야 
                                  #include_top = False (input layer 쓰지 않겠다)
                                  #대신 input은 3차원으로 줘야 하는 듯 CNN 쓰셨나...
                                  #색깔도... 컬러여야 하는 듯... channel=3이어야 한다


vgg16.trainable=False #학습시키지 않겠다 이미지넷 가져다가 그대로 쓰겠다 
# model.trainable=True

vgg16.summary() 



print("동결한 후 훈련되는 가중치의 수: ", len(vgg16.trainable_weights)) #model.trainable=False 돼 있으면 안 나옴 #32 (LAYER 16 * (가중치 1개 + BIAS 1개) = 32)
# 동결하기 전 훈련되는 가중치의 수:  32
# 동결한 후 훈련되는 가중치의 수:  0



'''
model.trainable=False
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
model.trainable=True
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''


'''
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 100, 100, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 100, 100, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 100, 100, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 50, 50, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 50, 50, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 50, 50, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 25, 25, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 25, 25, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 25, 25, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 25, 25, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 12, 12, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 12, 12, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 12, 12, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 12, 12, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 6, 6, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 3, 3, 512)         0
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
maxpooling에서 끝나 있음. output까지 연결?
'''

#cifar-10이라면? 
#Traceback (most recent call last):
#   File "d:\Study\keras2\keras75_VGG16.py", line 96, in <module>
#     model.add(Dense(10, activation='softmax'))
# AttributeError: 'Functional' object has no attribute 'add'

#VGG 모델이 함수형인지 sequential인지 

model = Sequential()
model.add(vgg16) #가중치 2개
model.add(Flatten())
model.add(Dense(256)) #가중치 2개
# model.add(BatchNormalization()) #가중치 연산을 함 (가중치 6 -> 8)
# model.add(Dropout(0.2)) #가중치 연산을 안함 (가중치 6 -> 6)
model.add(Activation('relu')) #가중치 연산을 안함 (가중치 6 -> 6)
model.add(Dense(256)) #가중치 2개 
model.add(Dense(10, activation='softmax'))

model.summary()
print("동결하기 전 훈련되는 가중치의 수: ", len(model.trainable_weights)) #layer 가중치 1개 + BIAS 1개 = 2
# print(model.trainable_weights)

import pandas as pd
pd.set_option('max_colwidth',-1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type','Layer Name','Layer Trainable'])
print(aaa.loc[:])
'''
0  <tensorflow.python.keras.engine.functional.Functional object at 0x00000185DBCF9580>  vgg16       False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x00000185DBD0FB80>           flatten     True
2  <tensorflow.python.keras.layers.core.Dense object at 0x00000185DBD7C490>             dense       True
3  <tensorflow.python.keras.layers.core.Activation object at 0x00000185DBD943D0>        activation  True
4  <tensorflow.python.keras.layers.core.Dense object at 0x00000185DBD9E220>             dense_1     True
5  <tensorflow.python.keras.layers.core.Dense object at 0x00000185E86F8C10>             dense_2     True
'''

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 1, 1, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 10)                5130
=================================================================
Total params: 14,719,818
Trainable params: 5,130
Non-trainable params: 14,714,688
_________________________________________________________________
trainable params: 5130 ??? 
flatten 512 + 1 = 513    -> output 10 -> 513 * 10 = 5130 
위에서 flatten해서 던져 주고 출력 model 10이니까 513 * 10 
'''

'''
과적합 줄이기
훈련데이터 줄인다
feature수를 줄인다
정규화 (batch normalization)/ Dropout 
'''