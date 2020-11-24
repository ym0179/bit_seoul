#Day12
#2020-11-24

# 당뇨병 데이터
# 모델 : RandomForestRegressor
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_diabetes(return_X_y=True)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

params = [
    {'n_estimators' : [300, 400, 500], #결정 트리의 개수, default=10, 많을 수록 좋은 성능이 나올 "수"도 있음 (시간이 오래걸림)
    'max_depth' : [6, 8, 10], #트리의 깊이, default=None(완벽하게 클래스 값이 결정될 때 까지 분할), 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
    'min_samples_leaf' : [7, 10, 12, 14], #리프노드가 되기 위한 최소한의 샘플 데이터 수, default=2, min_samples_split과 함께 과적합 제어 용도
    'min_samples_split' : [12, 14, 16], #노드를 분할하기 위한 최소한의 데이터 수, default=2, 과적합을 제어하는데 사용 (작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가)
    'n_jobs' : [-1]} #모든 코어를 다 쓰겠다
]

model = RandomizedSearchCV(RandomForestRegressor(), params, cv=kfold, verbose=2)


# 3. 훈련
model.fit(x_train,y_train) #model: RandomizedSearchCV


# 4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

# RandomizedSearchCV refit으로 이미 학습이 된 estimator 반환
estimator = model.best_estimator_

y_predict = estimator.predict(x_test)
print("(테스트 데이터 세트 r2) 최종정답률 : ", r2_score(y_test,y_predict))

'''
최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=12, 
min_samples_split=12,
                      n_estimators=400, n_jobs=-1)
최적 하이퍼 파라미터 :  {'n_jobs': -1, 'n_estimators': 400, 'min_samples_split': 12, 'min_samples_leaf': 12, 'max_depth': 6}
최고 정확도 : 0.4409
(테스트 데이터 세트 r2) 최종정답률 :  0.4142511040047415
'''