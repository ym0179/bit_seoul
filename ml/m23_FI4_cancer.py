#Day13
#2020-11-25

#xgboost
#1. FI 0 제거 또는 2. 하위 30% 제거
#3. 디폴트랑 성능 비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

x,y = load_breast_cancer(return_X_y=True)

x = pd.DataFrame(x)
x = x.drop([0,9,25],axis=1)
# print(x)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(455, 30)

model = XGBClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

print(model.feature_importances_)
# [0.         0.03518598 0.00053468 0.02371635 0.00661651 0.02328466
#  0.00405836 0.09933352 0.00236719 0.         0.01060954 0.00473884
#  0.01074011 0.01426315 0.0022232  0.00573987 0.00049415 0.00060479
#  0.00522006 0.00680739 0.01785728 0.0190929  0.3432317  0.24493258
#  0.00278067 0.         0.01099805 0.09473949 0.00262496 0.00720399]

'''
default
acc :  0.9736842105263158

FI 0인 컬럼(0,9,25) 제거
acc :  0.9736842105263158
'''