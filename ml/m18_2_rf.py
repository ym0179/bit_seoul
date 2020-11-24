#Day12
#2020-11-24

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x,y = load_breast_cancer(return_X_y=True)
cancer = load_breast_cancer()

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
# print(x_train.shape) #(455, 30)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc) #acc :  0.9210526315789473

print(model.feature_importances_)
'''
=> 중요 피쳐가 달라짐
*DecisionTree
acc :  0.9298245614035088
[0.         0.0624678  0.         0.         0.00738884 0.
 0.         0.         0.         0.         0.         0.
 0.         0.01297421 0.         0.         0.         0.
 0.         0.02364429 0.         0.01695087 0.         0.75156772  
 0.         0.         0.         0.12008039 0.         0.00492589]

*RF
 acc :  0.9649122807017544
[0.02021216 0.0123936  0.06021996 0.07598544 0.00425346 0.00432588  
 0.05630623 0.10240857 0.00213134 0.00258453 0.02356984 0.00203201  
 0.01026052 0.02229213 0.00179249 0.00346305 0.00502229 0.00173644  
 0.00449625 0.00279076 0.13364575 0.02315148 0.14512739 0.0977942   
 0.00780936 0.01409214 0.01335892 0.13523461 0.00524553 0.00626365] 
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