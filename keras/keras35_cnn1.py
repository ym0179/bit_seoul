#Day6
#2020-11-16

#CNN 조각조각 잘라서 특성값을 빼냄

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #Conv2D 2차원 이미지를 다루는 컨볼루션 레이어 클래스
# from keras.layers도 되는데 시간차 살짝 남 (살짝 느림) 

# feature 특성을 추출하기 때문에 Conv layer는 여러번 쌓을수록 좋아짐
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) #channels = 흑백: 1 / 칼라: 3, 10X10 이미지
# 2x2 = fliter / kernal size
# 입력모양 : batch_size, rows, cols, channels # 모델 관점에서 볼 때
# batch_size: 전체 데이터 안에서 내가 "학습에" 이용할 데이터 전체 개수 = 1 iteration에 학습할 데이터 개수
# iteration: 나뉘어진 데이터 조각 하나를 학습 / epoch는 모든 데이터 셋을 한번 학습한 횟수
# input_shape = (rows, cols, channels) #batch_size 뺀 나머지
'''
tf.keras.layers.Conv2D(
    filters, # 필터의 개수. 많을수록 많은 특징을 추출할 수 있지만, 학습 속도가 느리고 과적합 문제 발생
    #fliter수 = output_fliter_size
    kernel_size, # 필터 행렬의 크기를 결정
    strides=(1, 1), # 필터가 계산 과정에서 한 스텝마다 이동하는 크기 = Image에 Filter(Kernel)적용 시, 몇칸을 이동하며 Convolution feature를 얻을지에 대한 크기 정보
    # (디폴트 = 1)
    padding="valid", # Padding값을 조절해 Input size와 Output size 같게 만들기 (없으면 output size가 작아짐) = 연산 전에 주변에 빈 값을 넣어서 이미지의 크기를 유지
    # 2가지 옵션 : 'valid' 값은 비활성화, 'same'값은 빈 값을 넣어서 입력과 출력의 크기가 같도록 함 (상하좌우 가상의 공간 1을 설정, Input과 같은 size의 Convolution feature 얻음)
    # padding="same" 이 zero padding => 빈 값이 0인 경우 zero padding이라고함, layer마다 지정
    # 가장자리 부분에 데이터 손실을 방지 -> 가장자리는 특성이 한번 밖에 추출이 안됨
    # (디폴트 = 'valid')
)
'''
#참고 LSTM
#units
#reutrn_sequence
#입력모양: batch_size, timesteps, feature # 모델 관점에서 볼 때
#input_shape = (timesteps, feature) #batch_size 뺀 나머지, feature: 몇개씩 자르는지

# model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # output 값: (9x9x10) 
model.add(Conv2D(5, (2,2), padding='same')) #output 값: (8x8x5) #padding 넣어줬을 때 output: (9x9x5)
model.add(Conv2D(3, (3,3), padding='valid')) #output 값: (6x6x3) #padding 안 넣어줬을 때 output: (7x7x3) (input size 9x9x5)
model.add(Conv2D(7, (2,2))) #output 값: (6x6x7)
# conv는 layer output에서 차원이 안바뀜 -> 바로 Dense로 못 감
# LSTM은 두개 엮으면 한 차원 줄어듬 -> 바로 Dense

from tensorflow.keras.layers import MaxPooling2D
# Maxpolling 2D
# 이미지 자를 때 **중복하지 않고 최대값 (특징)만 추출 -> 특성이 가장 높은거만 추출
# 가장 특성치가 높은거만 남기고 데이터를 확 줄여줌
# 사소한 픽셀의 값을 무시하고, 가장 큰 특징을 나타내는 값을 기록하는 방식
model.add(MaxPooling2D()) #pool_size default 2 -> output 크기 반으로 줌 -> output 값: (3x3x7) (input size 6x6x3이었음)
# pool_size: 연산 범위를 의미. 해당 범위 내의 가장 큰 수만을 가져옴


'''
플래튼 레이어(Flatten Layer)
- CNN에서 컨볼루션 레이어와 풀링 레이어를 반복적으로 거치면 주요 특징만 추출됨 
- 추출된 주요 특징은 2차원 데이터로 이루어져 있음
- Dense와 같이 분류를 위한 학습 레이어에서는 1차원 데이터로 바꾸어서 학습 해야함
tf.keras.layers.Flatten(): 2차원 데이터를 1차원 데이터로 바꾸는 역할의 레이어
'''
from tensorflow.keras.layers import Flatten
model.add(Flatten())
# input: (3x3x7)
# output: 3*3*7 = 63 -> (63, )
model.add(Dense(1))

model.summary()
# 2 x 2(필터 크기) * 1 (입력 채널(RGB)) * 10(출력 채널) + 10(출력 채널 bias) / (1(입력) * 2 x 2 (커널 사이즈) + 1(bias)) * 10(출력)
# 2 x 2(필터 크기) * 10 (입력 채널) * 5(출력 채널) + 5(출력 채널 bias)
# 3 x 3(필터 크기) * 5 (입력 채널) * 3(출력 채널) + 3(출력 채널 bias)
# 2 x 2(필터 크기) * 3 (입력 채널) * 7(출력 채널) + 7(출력 채널 bias)
# Dense layer: ( 63(입력 node) + 1(bias) ) * 1(출력 node) = 4

