#Day12
#2020-11-24

# 유방암 데이터
# 모델 : RandomForestClassifier
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_breast_cancer(return_X_y=True)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

# grid search; key, value 쌍의 데이터를 입력 받아 key값에 해당하는 파라미터의 값을 튜닝 시켜줌
# random forest hyperparameter
params = [
    {'n_estimators' : [100, 200], #결정 트리의 개수, default=10, 많을 수록 좋은 성능이 나올 "수"도 있음 (시간이 오래걸림)
    'max_depth' : [6, 8, 10, 12], #트리의 깊이, default=None(완벽하게 클래스 값이 결정될 때 까지 분할), 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
    'min_samples_leaf' : [3, 5, 7, 10], #리프노드가 되기 위한 최소한의 샘플 데이터 수, default=2, min_samples_split과 함께 과적합 제어 용도
    'min_samples_split' : [2, 3, 5, 10], #노드를 분할하기 위한 최소한의 데이터 수, default=2, 과적합을 제어하는데 사용 (작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가)
    'n_jobs' : [-1]} #모든 코어를 다 쓰겠따
]

model = GridSearchCV(RandomForestClassifier(), params, cv=kfold)


# 3. 훈련
model.fit(x_train,y_train) #model: GridSearch


# 4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
model = model.best_estimator_

y_predict = model.predict(x_test)
print("(테스트 데이터 세트 정확도) 최종정답률 : ", accuracy_score(y_test,y_predict))

'''
최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=-1)
최적 하이퍼 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}
최고 정확도 : 0.9582
(테스트 데이터 세트 정확도) 최종정답률 :  0.9736842105263158
'''