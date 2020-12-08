#Day22
#2020-12-08

#python lib pickle로 모델 저장

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x,y = load_breast_cancer(return_X_y=True)
# cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
# print(x_train.shape) #(455, 30)

model = XGBClassifier(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc) #acc :  0.9736842105263158

import pickle
#모델 저장
#모델 + 가중치
pickle.dump(model, open("./save/xgb_save/cancer.pickle.dat","wb"))
print("saved successfully")

model2 = pickle.load(open("./save/xgb_save/cancer.pickle.dat","rb"))
print("loaded successfully")
acc2 = model2.score(x_test,y_test)
print("acc2 : ", acc2) #acc2 :  0.9736842105263158