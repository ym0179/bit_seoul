#Day20
#2020-12-04

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

model = ResNet50()
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) 

model.trainable=True 
model.summary() 

print("동결한 후 훈련되는 가중치의 수: ", len(model.trainable_weights)) 
#VGG16 => 32 (LAYER 16 * (가중치 1개 + BIAS 1개) = 32)

'''
모델별로 가장 순수했을 떄의 파라미터의 갯수와 가중치 수를 정리하시오.

VGG16 / 138,357,544 / 32
VGG19 / 143,667,240 / 38
Xception / 22,910,480 /156
ResNet101 / 44,707,176 / 418
ResNet101V2 / 44,675,560 / 344
ResNet152 / 60,419,944 / 622
ResNet152V2 / 60,380,648 / 514
ResNet50 / 25,636,712 / 214
ResNet50V2 / 25,613,800 / 174
InceptionResNetV2 / 55,873,736 / 490
InceptionV3 / 23,851,784 / 190
MobileNet / 4,253,864 / 83
MobileNetV2 / 3,538,984 / 158
DenseNet121 / 8,062,504 / 364
DenseNet169 / 14,307,880 / 508
DenseNet201 / 20,242,984 / 604
NASNetLarge / 88,949,818 / 1018
NASNetMobile / 5,326,716 / 742
'''

