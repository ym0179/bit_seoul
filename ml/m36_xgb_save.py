#Day22
#2020-12-08

#xgb save_model로 모델 저장

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


# import pickle
# pickle.dump(model, open("./save/xgb_save/cancer.pickle.dat","wb"))
# import joblib
# joblib.dump(model, "./save/xgb_save/cancer.joblib.dat")

model.save_model("./save/xgb_save/cancer.xgb.model")
#모델 저장 - 모델 + 가중치
print("saved successfully")

# model2 = pickle.load(open("./save/xgb_save/cancer.pickle.dat","rb"))
# model2 = joblib.load("./save/xgb_save/cancer.joblib.dat")
model2 = XGBClassifier() #xgbclassifier 명시
model2.load_model("./save/xgb_save/cancer.xgb.model")
print("loaded successfully")

acc2 = model2.score(x_test,y_test)
print("acc2 : ", acc2) #acc2 :  0.9736842105263158