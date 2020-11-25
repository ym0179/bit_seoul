#Day13
#2020-11-25

#xgboost
#1. FI 0 제거 또는 2. 하위 30% 제거
#3. 디폴트랑 성능 비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x,y = load_iris(return_X_y=True)
iris = load_iris()

x = x[:,1:]

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(120, 4)

model = XGBClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

print(model.feature_importances_)
# [0.01759811 0.02607087 0.6192673  0.33706376]


'''
default
acc :  0.9

FI 제일 하위인 0번째 column 제거
acc :  0.9
'''