#Day13
#2020-11-25

#과적합 방지
#1. 훈련데이터량을 늘린다.
#2. 피처수를 줄인다.
#3. regularization

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV


x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

params = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

model = RandomizedSearchCV(XGBRegressor(), params, n_jobs=n_jobs, cv=kfold, verbose=2)

model.fit(x_train,y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

r2 = model.score(x_test,y_test)
print("r2 : ", r2)

'''
최적 하이퍼 파라미터 :  {'n_estimators': 110, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
최고 정확도 : 0.8826
r2 :  0.9062926224740284
'''