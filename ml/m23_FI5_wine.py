#Day13
#2020-11-25

#xgboost
#1. FI 0 제거 또는 2. 하위 30% 제거
#3. 디폴트랑 성능 비교
#winequality-white.csv

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust - 이상치 제거에 효과
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#pandas로 csv 불러오기
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')

#x,y 값 나누기
y = wine['quality']
x = wine.drop('quality',axis=1)

# x = x.drop(x.columns[[0,4,8]], axis=1)
# print(x)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(3918, 11)

model = XGBClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

print(model.feature_importances_)
# [0.06218192 0.11186972 0.07586386 0.07901246 0.0632134  0.08535022
#  0.06725179 0.06797317 0.06311163 0.07061544 0.25355643]

# 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()

'''
default
acc :  0.6551020408163265

FI 0인 컬럼(0,4,8) 제거
acc :  0.636734693877551
'''