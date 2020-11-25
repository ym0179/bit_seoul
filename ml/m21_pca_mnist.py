#Day13
#2020-11-25

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

x = x.reshape(-1, 28*28) #reshape(-1, 정수)
print(x.shape) #(70000, 784)


### 실습
# pca 사용 0.95 이상인게 몇개?

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
print('선택할 차원 수 :', d)
# 선택할 차원 수 : 154

# 그래프로 그리기
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

# n_components에 0 ~ 1사이의 값을 지정해 PCA 계산을 할 수도 있음
pca = PCA(n_components=0.95) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
X_reduced = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)
# 선택한 차원(픽셀) 수 : 154

'''
* reshape(-1, 정수)
총 12개의 원소가 들어있는 배열 x에 대해서 x.reshape(-1, 정수) 를 해주면 
'열(column)' 차원의 '정수'에 따라서 12개의 원소가 빠짐없이 배치될 수 있도록 
'-1'이 들어가 있는 '행(row)' 의 개수가 가변적으로 정해짐
'''