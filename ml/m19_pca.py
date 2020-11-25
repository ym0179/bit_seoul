#Day13
#2020-11-25

'''
PCA 주성분분석(Principal Component Analysis)
=> 여러 변수 간 존재하는 상관관계를 이용해 이를 대표하는 주성분을 추출해 차원을 축소
- 차원축소 (dimensionality reduction/dimension reduction)와 변수추출 (feature extraction) 기법으로 널리 쓰임

피처 추출 (Feature extraction) : 기존 피처를 저차원의 중요 피처로 압축하여 추출하는 것 (=> 기존 피처와는 완전히 다른 값이 되어버림)
- 피처가 많을 경우 개별 피처간 상관 관계가 높을 가능성이 커짐
- 선형 모델에서는 입력한 변수들간의 상관관계가 높을 경우에 이로 인한 다중 공선성 문제 발생 (=> 모델의 성능 저하...)

다중공선성: 회귀분석에서 사용된 모형의 설명 변수가 다른 설명 변수와 상관 정도가 높아 데이터 분석시 부정적인 영향을 미치는 현상
-> 회귀분석에서는 설명 변수들을 모두 일정하다고 생각하고 검증함 (서로 독립이라고 가정)
-> 하지만 두 변수가 서로에게 영향을 주고 있으면 둘 중 하나의 영향력을 검증 할 때 다른 하나의 영향력을 완벽히 통제할 수 없음
-> 다중공선성 문제 확인하는 방법은 상관관계 분석을 통해 변수 간 상관도 분석하거나 회귀분석의 공선성 확인(?)
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA(n_components=7) #축소 후 칼럼의 개수
x2d = pca.fit_transform(x)
print(x2d.shape) #(442, 7)

pca_EVR = pca.explained_variance_ratio_ #PCA 컴포넌트별로 차지하는 변동성 비율을 확인
print(pca_EVR) #[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
print(sum(pca_EVR)) #0.9479436357350411
# PCA를 7개 요소로 변환해도 원본 데이터의 변동성을 약 94.8% 설명할 수 있다는 뜻