#Day18
#2020-12-02

import numpy as np
import matplotlib.pyplot as plt

# ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
def relu(x):
    return np.maximum(0,x) #0이하는 0, 0이상은 x

x = np.arange(-5,5,0.1)
y = relu(x)

# print("x : ", x)
# print("y : ", y)

plt.plot(x,y)
plt.grid()
plt.show()

# relu 친구들 찾기
# relu 계열
 
# ReLU의 뉴런이 죽는("Dying ReLu")현상을 해결하기위해 나온 함수 (단점 보완)
# Leakly ReLU는 음수의 x값에 대해 미분값이 0되지 않는다는 점을 제외하면 ReLU와 같은 특성을 가짐
def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return np.maximum(0.01*x,x) # 일반적으로 알파를 0.01로 설정

plt.plot(x, leakyrelu_func(x), linestyle='--', label="Leaky ReLU")
 
# 비교적 가장 최근에 나온 함수
# "Dying ReLU" 문제를 해결
# 출력값이 거의 zero-centered에 가까움
# 일반적인 ReLU와 달리 exp함수를 계산하는 비용이 발생
# 지수 함수를 이용하여 입력이 0 이하일 경우 부드럽게 깎아줌
#  별도의 알파 값을 파라미터로 받는데 일반적으로 1로 설정
def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*1*(np.exp(x)-1)
 
plt.plot(x, elu_func(x), linestyle='--', label="ELU(Exponential linear unit)")

def selu_func(x,a): # SELU(Scaled Exponential Linear Unit)
    return (x>=0)*x + (x<0)*a*(np.exp(x)-1)
 
 
def trelu_func(x): # Thresholded ReLU
    return (x>1)*x # 임계값(1) 조정 가능
 
plt.plot(x, trelu_func(x), linestyle='--', label="Thresholded ReLU")

plt.show()

# https://keras.io/ko/layers/advanced-activations/