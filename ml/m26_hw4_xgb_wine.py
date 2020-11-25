#Day13
#2020-11-25

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')

#x,y 값 나누기
y = wine['quality']
x = wine.drop('quality',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

params = [
    {"xgb__n_estimators":[100,200,300], "xgb__learning_rate":[0.1,0.3,0.01,0.001], "xgb__max_depth":[4,5,6]},
    {"xgb__n_estimators":[90,100,110], "xgb__learning_rate":[0.1,0.01,0.001], "xgb__max_depth":[4,5,6], "xgb__colsample_bytree":[0.6,0.9,1]},
    {"xgb__n_estimators":[9,110], "xgb__learning_rate":[0.1,0.001,0.5], "xgb__max_depth":[4,5,6], "xgb__colsample_bytree":[0.6,0.9,1], "xgb__colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

pipe = Pipeline([("scaler", MinMaxScaler()),('xgb', XGBClassifier())])

model = RandomizedSearchCV(pipe, params, cv=5, n_jobs=n_jobs, verbose=2)

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

acc = model.score(x_test,y_test)
print("acc : ", acc)

'''
RobustScaler
최적 하이퍼 파라미터 :  {'xgb__n_estimators': 110, 'xgb__max_depth': 5, 'xgb__learning_rate': 0.5, 'xgb__colsample_bytree': 1, 'xgb__colsample_bylevel': 0.7}
최고 정확도 : 0.6445
acc :  0.6632653061224489

StandardScaler
최적 하이퍼 파라미터 :  {'xgb__n_estimators': 110, 'xgb__max_depth': 6, 'xgb__learning_rate': 0.1, 'xgb__colsample_bytree': 0.9}
최고 정확도 : 0.6284
acc :  0.6612244897959184

MinMaxScaler
최적 하이퍼 파라미터 :  {'xgb__n_estimators': 9, 'xgb__max_depth': 6, 'xgb__learning_rate': 0.5, 'xgb__colsample_bytree': 0.9, 'xgb__colsample_bylevel': 0.6}
최고 정확도 : 0.5972
acc :  0.6357142857142857

MaxAbsScaler
최적 하이퍼 파라미터 :  {'xgb__n_estimators': 110, 'xgb__max_depth': 5, 'xgb__learning_rate': 0.5, 'xgb__colsample_bytree': 0.6, 'xgb__colsample_bylevel': 0.6}
최고 정확도 : 0.6536
acc :  0.6642857142857143
'''