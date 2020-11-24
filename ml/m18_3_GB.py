#Day12
#2020-11-24

# Gradient Boost
# boosting 계열의 앙상블 알고리즘 - 단순하고 약한 학습기(Weak Learner)를 결합하고 틀린 것에 가중치를 부여해서 보다 정확하고 강력한 학습기(Strong Learner)를 만드는 방식
# 정확도가 낮더라도 일단 모델을 만들고, 드러난 약점(예측 오류)은 두 번째 모델이 보완 (남아 있는 문제(오차)를 다음 모델에서 보완하여 계속 더하는 과정을 반복)
# 배깅의 대표적인 모델은 랜덤 포레스트
# 가중치를 부여하는 방식이 Gradient Descent

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# x,y = load_breast_cancer(return_X_y=True)
cancer = load_breast_cancer()

# train-test split
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, train_size=0.8)
# print(x_train.shape) #(455, 30)

model = GradientBoostingClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc) #acc :  0.9210526315789473

print(model.feature_importances_)
'''
acc :  0.956140350877193
[1.63371983e-03 5.66761193e-02 4.38040549e-04 2.06139501e-04
 2.97728400e-03 2.22887898e-03 6.44353279e-04 8.82164391e-02
 1.61658417e-03 6.42735579e-04 3.74587880e-03 4.75734192e-04
 1.27279996e-05 1.28082760e-02 2.19452127e-03 3.48930232e-03
 2.67986916e-03 5.42349859e-04 3.19316993e-04 2.66452733e-03
 2.29156957e-01 3.32686797e-02 8.28373754e-03 4.18977259e-01
 2.57976083e-03 6.64772194e-04 8.80605295e-03 1.12006363e-01
 8.67928200e-05 1.95682570e-03]
'''

# 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()