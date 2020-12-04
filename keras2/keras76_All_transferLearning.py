#Day20
#2020-12-04

from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

model = ResNet101V2()
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
ResNet101
ResNet101V2
'''

