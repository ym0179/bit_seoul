#Day22
#2020-12-08

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import pickle

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

model = XGBRegressor(n_jobs=-1)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("R2: ", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

best_score = 0.0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #파라미터 더 있음
    
    select_x_train = selection.transform(x_train)
    
    selection_model =  XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score))

    #모델 + 가중치 저장
    pickle.dump(model, open("./save/xgb_save/m37_boston.pickle." + str(np.round_(score,2)) + ".dat","wb"))
    # model.save_model("./save/xgb_save/m37_boston.pickle." + str(np.round_(score,2)) + ".dat")
    print("saved successfully")

    if best_score < score:
        best_score = score
        pickle.dump(model, open('./save/xgb_save/m37_boston_best.pickle.dat','wb'))

model2 = pickle.load(open("./save/xgb_save/m37_boston_best.pickle.dat","rb"))
print("loaded successfully")
r2 = model2.score(x_test,y_test)
print("R2 : ", r2)
