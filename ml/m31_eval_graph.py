#Day21
#2020-12-07

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

# 1. 데이터
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

# 2. 모델
model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
# model = XGBRegressor(learning_rate=0.01) #n_estimators default = 100

# 3. 훈련
model.fit(x_train,y_train,
          verbose=True, # verbose 0또는 1
          eval_metric=['rmse','logloss'],
          eval_set=[(x_train,y_train),(x_test,y_test)], #훈련, 평가 같이 보기
        #   early_stopping_rounds=20
          )

# 평가지표
# https://xgboost.readthedocs.io/en/latest/parameter.html
# rmse, mae, rmsle
#

# 4. 평가, 예측
results = model.evals_result()
# print("eval's results : ",results) #dict 형태
# print(results.keys()) #dict_keys(['validation_0', 'validation_1'])
print("eval's results [rmse]: ", results['validation_1']['rmse'][-1]) #마지막 rmse 값
print("eval's results [logloss]: ", results['validation_1']['logloss'][-1]) #마지막 logloss 값


y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred) 
print("r2 : ", r2) #r2 :  0.9018263688695799

score = model.score(x_test,y_test)
print("R2: ", score) #R2:  0.9018263688695799

#그래프
import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0,epochs)

fig,ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label="Train")
ax.plot(x_axis, results['validation_1']['logloss'], label="Test")
ax.legend()
plt.ylabel("Log Loss")
plt.title("XGBoost Log Loss")

fig,ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label="Train")
ax.plot(x_axis, results['validation_1']['rmse'], label="Test")
ax.legend()
plt.ylabel("Rmse")
plt.title("XGBoost Rmse")
plt.show()
