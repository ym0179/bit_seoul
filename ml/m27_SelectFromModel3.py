#Day14
#2020-11-26

# SelectFromModel은 (지도 학습 모델로 계산된) 중요도가 지정한 임계치보다 큰 모든 특성을 선택
# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치 적용
# 최적의 R2값과 피처임포턴츠 구할 것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서 그리드서치 또는 랜덤서치 적용해서 최적의 R2값 구할 것

# 1번값과 2번값 비교

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)

params = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

model = RandomizedSearchCV(XGBRegressor(), params, n_jobs=n_jobs, cv=5)

model.fit(x_train,y_train)

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

r2 = model.score(x_test,y_test)
print("r2 : ", r2)

thresholds = np.sort(model.feature_importances_)
# print(thresholds)

save_score = 0
best_thresh = 0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    
    selection_model =  XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)

    print("Thresh=%.4f, n=%d, R2: %.4f%%" %(thresh, select_x_train.shape[1], score))

    if score > save_score:
        save_score = score
        best_thresh = thresh
    # print("best_thresh, save_score: ", best_thresh, save_score)

print("best_thresh, save_score: ", best_thresh, save_score)

selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
x_train = selection.transform(x_train)
x_test = selection.transform(x_test)

model = RandomizedSearchCV(XGBRegressor(), params, n_jobs=n_jobs, cv=5)

model.fit(x_train,y_train)

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

r2 = model.score(x_test,y_test)
print("r2 : ", r2)

'''
최적 하이퍼 파라미터 :  {'n_estimators': 90, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
최고 정확도 : 0.3667
r2 :  0.45565412996943155

Thresh=0.0395, n=10, R2: 0.3839%
Thresh=0.0402, n=9, R2: 0.3455%
Thresh=0.0474, n=8, R2: 0.3340%
Thresh=0.0530, n=7, R2: 0.3027%
Thresh=0.0564, n=6, R2: 0.3421%
Thresh=0.0721, n=5, R2: 0.3458%
Thresh=0.0727, n=4, R2: 0.3018%
Thresh=0.0736, n=3, R2: 0.3432%
Thresh=0.1887, n=2, R2: 0.2736%
Thresh=0.3565, n=1, R2: -0.0518%

best_thresh, save_score:  0.039451897 0.3838801478911562

최적 하이퍼 파라미터 :  {'n_estimators': 110, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1, 'colsample_bylevel': 0.7}
최고 정확도 : 0.3754
r2 :  0.4733078876221164
'''