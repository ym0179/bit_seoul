#Day12
#2020-11-24

# 의사결정나무 / 결정트리 / Decision Tree : 분류와 회귀 모두 가능한 지도학습 모델
# 결정 트리는 스무고개 하듯이 예/아니오 질문을 이어가며 학습
# 특정 기준(질문)에 따라 데이터를 구분하는 모델
# 질문이나 정답을 담은 네모 상자를 노드(Node)라고 함
# 맨 처음 분류 기준 (즉, 첫 질문)은 Root Node, 맨 마지막 노드는 Terminal Node / Leaf Node

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x,y = load_breast_cancer(return_X_y=True)
cancer = load_breast_cancer()

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(455, 30)

model = DecisionTreeClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc) #acc :  0.9210526315789473

print(model.feature_importances_)
# 필요없는 피쳐들
# 속도 느려짐, 자원 낭비...
'''
acc :  0.9298245614035088
[0.         0.0624678  0.         0.         0.00738884 0.
 0.         0.         0.         0.         0.         0.
 0.         0.01297421 0.         0.         0.         0.
 0.         0.02364429 0.         0.01695087 0.         0.75156772  
 0.         0.         0.         0.12008039 0.         0.00492589]
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