#Day13
#2020-11-25

# PCA
# 적절한 차원 수 선택하는 법!! 누적된 분산의 비율이 95%가 되는 주성분 축, 차원을 선택하는 것

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
print('선택할 차원 수 :', d)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

# 그래프로 그리기
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

# n_components에 0 ~ 1사이의 값을 지정해 PCA 계산을 할 수도 있음
pca = PCA(n_components=0.95) #데이터셋에 분산의 95%만 유지하도록 PCA를 적용
X_reduced = pca.fit_transform(x)
print('선택한 차원(픽셀) 수 :', pca.n_components_)
