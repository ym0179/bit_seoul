#Day3
#2020-11-11
'''
<함수형 모델>
지금까지 사용한 모델은 Sequential 모델
-> 여러 층을 공유하거나 다양한 종류의 입출력을 사용하는 등의 복잡한 모델을 만들기에는 한계가 존재
-> 복잡한 모델을 생성할 수 있는 함수형 모델 (Functional API Model)
=> 간단한 구조를 만들때에는 Sequential API를 이용해 직관적이고 빠르게 모델을 만들고
   복잡한 구조의 모델을 만들 때에는 주로 Functional API 사용


<활성화 함수>
딥러닝에서 각 노드에서 연산된 가중치가 곱셈을 통해 다음 레이어로 전달이 됨
딥러닝 네트워크에서는 노드에 들어오는 값들에 대해 곧바로 다음 레이어로 전달하지 않고
주로 비선형 함수를 통과시킨 후 전달
이때 사용하는 함수를 활성화 함수(Activation Function)라고 부름
비선형 함수를 사용하는 이유는 선형함수를 사용할 시 층을 깊게 하는 의미가 줄어들기 때문
-> 선형함수인 h(x)=cx를 활성화함수로 사용한 3층 네트워크는 y(x)=h(h(h(x))) ==  y(x)=ax와 똑같은 식
-> 즉, 은닉층이 없는 네트워크로 표현할 수 있음
-> 딥러닝에서 층을 쌓는 혜택을 얻고 싶다면 활성화함수로는 반드시 비선형 함수를 사용

* 레이어 마다 default로 활성화 함수가 존재
* regressor/linear 회귀모델에서는 default로 'linear' 활성화 함수 씀
'''

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array(range(101,201))
x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


#2. 모델 구성
#함수 -> 저장했다가 다시 쓸 수 있음
# from keras.models import Sequential #조금 더 느림
from tensorflow.keras.models import Sequential, Model #함수형 모델
#함수형 모델은 layer에 input layer가 별도로 존재
from tensorflow.keras.layers import Dense, Input

#Functional Model 함수형 모델
input1 = Input(shape=(3,)) #입력층 정의
dense1 = Dense(5, activation='relu')(input1) #input layer연결
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #마지막 layer는 linear 활성화 함수여야함
model = Model(inputs=input1, outputs=output1) #마지막에 모델 정의

# Sequential Model 
# model = Sequential()
# model.add(Dense(5, input_shape = (3,), activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1))

model.summary() #모델 구조 확인
'''
<모델 구조>
- Layer(type): 레이어의 이름과 타입, 지정하고 싶을 때에는 Dense에 파라미터로 name="이름"
- Output Shape: (None,5) => None개의 행과 5개의 아웃풋 값, 데이터의 갯수는 계속 추가될 수 있기 때문에 
딥러닝 모델에서는 행을 무시하고 열의 shape을 맞춰줌
- Param: 파라미터의 수, 입력노드와 출력노드에 연결된 간선의 수
    *y=wx+b
    *bias는 모든 layer에 존재
    *(노드의 수 + bias(=1)) * (input의 차원)

<param 계산>
3(입력) -> 5 -> 4 -> 3 -> 1(출력)
input layer - hidden layer1: (3+1)*5 = 20
hidden layer1 - hidden layer2: (5+1)*4 = 24
hidden layer2 - hidden layer3: (4+1)*3 = 15
hidden layer3 - output layer: (3+1)*1 = 4
'''
