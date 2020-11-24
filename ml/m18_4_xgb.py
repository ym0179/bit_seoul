#Day12
#2020-11-24

#Gradient Boost
#boosting 계열의 앙상블 알고리즘 - 약한 분류기를 결합하여 강한 분류기를 만드는 과정

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x,y = load_breast_cancer(return_X_y=True)
cancer = load_breast_cancer()

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
# print(x_train.shape) #(455, 30)

model = XGBClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc) #acc :  0.9210526315789473

print(model.feature_importances_)
'''
 acc :  0.9736842105263158
[0.         0.03518598 0.00053468 0.02371635 0.00661651 0.02328466        
 0.00405836 0.09933352 0.00236719 0.         0.01060954 0.00473884        
 0.01074011 0.01426315 0.0022232  0.00573987 0.00049415 0.00060479        
 0.00522006 0.00680739 0.01785728 0.0190929  0.3432317  0.24493258        
 0.00278067 0.         0.01099805 0.09473949 0.00262496 0.00720399]
'''

# 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()