#Day13
#2020-11-25

#xgboost
#1. FI 0 제거 또는 2. 하위 30% 제거
#3. 디폴트랑 성능 비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

x,y = load_boston(return_X_y=True)

x = x[:,[0,2,4,5,6,7,8,9,10,11,12]]

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(404, 13)

model = XGBRegressor(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

print(model.feature_importances_)
# [0.01669537 0.00150525 0.02149532 0.0007204  0.05927434 0.29080436
#  0.01197547 0.05330402 0.0360474  0.02261044 0.07038534 0.01352609
#  0.40165624]

'''
default
acc :  0.9328109815565079

FI 제일 하위 column (1,3)번째 제거
acc :  0.9292947621215256
'''