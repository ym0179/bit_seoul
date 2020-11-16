#Day6
#2020-11-16

#mnist (0-9까지의 손글씨) 예제

import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)
print(x_train[-1])
print(y_train[-1])

plt.imshow(x_train[-1], 'gray') #이미지 확인
plt.show()

'''
**분류문제에서 주의해야할 점
우리가 궁극적으로 하고자 하는 바는 0 ~ 255 사이의 값으로 이뤄진 원본 데이터를 보고, 이 데이터가 0 ~ 9 사이의 숫자 중 어떤 것인지 예측
- 0과 9는 동일한 값을 가져야한다  => 9는 1의 9배?? 동일한 경우의 수가 되어야함
    - 0 ~ 9까지의 정수 값을 갖는 형태가 아닌 0 이나 1로 이뤄진 벡터로 수정
    - 만약에 '3' 이라는 숫자라면 3을 [0, 0, 1, 0, 0, 0, 0, 0, 0] 로 바꿈

원-핫 인코딩(One-Hot Encoding): 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
레이블 인코딩(Lable encoding): 일괄적인 숫자 값으로 변환되면서 예측 성능이 떨어짐 = 숫자의 크고 작음에 대한 특성이 작용

'''

#원핫인코딩을 하면 (60000,) => (60000,10)으로 reshape (분류 10개)

#sklearn 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
ec = OneHotEncoder(y)
ec.fit_transform(y)

#케라스 원핫인코딩
#문자를 숫자로 바꿔준 후 (Label Encoding 또는 numeric -> cateogorical로 변환)
from tensorflow.keras.utils import np_utils
y = y.astype('category')
from sklearn.preprocessing import LabelEncoder
ec = LabelEncoder(y)
ec.fit_transform(y)

ec = np_utils.to_categorical(y)

#판다스 원핫인코딩
# DataFrame에서 category형 데이터 컬럼을 선택하여 자동으로 원핫인코딩을 해줌
# 만약 겉보기에는 수치형 데이터 컬럼이지만, 실제로는 categorical 컬럼이라면 이 역시 원핫인코딩을 해줌
import padas as pd
pd.get_dummies(y)