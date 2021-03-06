#Day12
#2020-11-24

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# 1. 데이터
x,y = load_boston(return_X_y=True)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

params = [
    {'rf__n_estimators' : [400, 500, 600], #결정 트리의 개수, default=10, 많을 수록 좋은 성능이 나올 "수"도 있음 (시간이 오래걸림)
    'rf__max_depth' : [6, 8, 10, 12, 14], #트리의 깊이, default=None(완벽하게 클래스 값이 결정될 때 까지 분할), 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
    'rf__min_samples_leaf' : [3, 5, 7, 10], #리프노드가 되기 위한 최소한의 샘플 데이터 수, default=2, min_samples_split과 함께 과적합 제어 용도
    'rf__min_samples_split' : [2, 3, 5, 10], #노드를 분할하기 위한 최소한의 데이터 수, default=2, 과적합을 제어하는데 사용 (작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가)
    'rf__n_jobs' : [-1]} #모든 코어를 다 쓰겠다
]

pipe = Pipeline([("scaler", RobustScaler()),('rf', RandomForestRegressor())])

model = RandomizedSearchCV(pipe, params, cv=5, verbose=2)

model.fit(x_train, y_train)

print('r2_score : ', model.score(x_test,y_test)) 
print('최적의 매개변수 : ', model.best_estimator_)
print('최적의 매개변수 : ', model.best_params_)

'''
r2_score :  0.9221367786453781
최적의 매개변수 :  Pipeline(steps=[('scaler', RobustScaler()),      
                ('rf',
                 RandomForestRegressor(max_depth=10, min_samples_leaf=3,
                                       n_estimators=300, n_jobs=-1))])
최적의 매개변수 :  {'rf__n_jobs': -1, 'rf__n_estimators': 300, 'rf__min_samples_split': 2, 'rf__min_samples_leaf': 3, 'rf__max_depth': 10}


r2_score :  0.9187597132816634
최적의 매개변수 :  Pipeline(steps=[('scaler', StandardScaler()),
                ('rf',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=3,
                                       n_estimators=200, n_jobs=-1))])
최적의 매개변수 :  {'rf__n_jobs': -1, 'rf__n_estimators': 200, 'rf__min_samples_split': 2, 'rf__min_samples_leaf': 3, 'rf__max_depth': 8}
'''
