#Day21
#2020-12-07

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

# 2. 모델
model = XGBClassifier(n_estimators=1000, learning_rate=0.01)

# 3. 훈련
model.fit(x_train,y_train,
          verbose=True, # verbose 0또는 1
          eval_metric=['logloss','error','auc'],
          eval_set=[(x_train,y_train),(x_test,y_test)] #훈련, 평가 같이 보기
          )

# 평가지표
# https://xgboost.readthedocs.io/en/latest/parameter.html
# error, logloss, auc

# 4. 평가, 예측
results = model.evals_result()
# print("eval's results : ",results) #dict 형태
# print(results.keys()) #dict_keys(['validation_0', 'validation_1'])
print("eval's results [logloss]: ", results['validation_1']['logloss'][-1]) #마지막 metrics 값
print("eval's results [error]: ", results['validation_1']['error'][-1]) #마지막 metrics 값
print("eval's results [auc]: ", results['validation_1']['auc'][-1]) #마지막 metrics 값


y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc) #acc:  0.9736842105263158

score = model.score(x_test,y_test)
print("acc: ", score) #acc:  0.9736842105263158
