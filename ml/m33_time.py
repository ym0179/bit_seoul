#Day22
#2020-12-08

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

model = XGBRegressor(n_jobs=-1)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("R2: ", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import time
start1 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #파라미터 더 있음
    
    select_x_train = selection.transform(x_train)
    
    selection_model =  XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train,verbose=0)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)
    # print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score))


start2 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #파라미터 더 있음
    
    select_x_train = selection.transform(x_train)
    
    selection_model =  XGBRegressor(n_jobs=6)
    selection_model.fit(select_x_train,y_train,verbose=0)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)
    # print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score))

end = start2 - start1
print("n_jobs = -1 걸린 시간 : ", end) #2.6997766494750977
end2 = time.time() - start2
print("n_jobs = 6 걸린 시간 : ", end2) #1.6067004203796387
