#Day12
#2020-11-24

# 파이프라인을 사용하면 데이터 사전 처리 및 분류의 모든 단계를 포함하는 단일 개체를 만들 수 있다.
# - cross validation 할 때 train 만 fit+transform하고 val는 transform만 해야된다 => 과적합 방지
# - 파이프라인을 사용하면 가능
# 학습할 때와 동일한 기반 설정으로 동일하게 테스트 데이터를 변환해야 함 (train dataset 만 fit) - 훈련 데이터의 분포 추정 : 훈련 데이터를 입력으로 하여 fit 메서드를 실행하여 분포 모수를 객체내에 저장
# 학습 데이터에서 Scale된 데이터를 기반으로 모델이 학습이 되었기 때문에 
# 이렇게 학습된 모델이 예측을 할 때에도 학습 데이터의 Scale 기준으로 테스트/검증 데이터를 변환 한 뒤 predict

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# 1. 데이터
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
#x,y 값 나누기
y = wine['quality']
x = wine.drop('quality',axis=1)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# '모델명__'이 파라미터 key값으로 들어가야함
params = [
    {'rf__n_estimators' : [100, 200], #결정 트리의 개수, default=10, 많을 수록 좋은 성능이 나올 "수"도 있음 (시간이 오래걸림)
    'rf__max_depth' : [6, 8, 10, 12], #트리의 깊이, default=None(완벽하게 클래스 값이 결정될 때 까지 분할), 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
    'rf__min_samples_leaf' : [3, 5, 7, 10], #리프노드가 되기 위한 최소한의 샘플 데이터 수, default=2, min_samples_split과 함께 과적합 제어 용도
    'rf__min_samples_split' : [2, 3, 5, 10], #노드를 분할하기 위한 최소한의 데이터 수, default=2, 과적합을 제어하는데 사용 (작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가)
    'rf__n_jobs' : [-1]} #모든 코어를 다 쓰겠다
]

pipe = Pipeline([("scaler", MaxAbsScaler()),('rf', RandomForestClassifier())])

model = RandomizedSearchCV(pipe, params, cv=5, verbose=2)

model.fit(x_train, y_train)

print('acc : ', model.score(x_test,y_test)) #acc :  1.0

print('최적의 매개변수 : ', model.best_estimator_)
print('최적의 매개변수 : ', model.best_params_)

'''
acc :  0.6795918367346939
최적의 매개변수 :  Pipeline(steps=[('scaler', MaxAbsScaler()),      
                ('rf',
                 RandomForestClassifier(max_depth=12, min_samples_leaf=3,
                                        min_samples_split=3, n_estimators=200,
                                        n_jobs=-1))])
최적의 매개변수 :  {'rf__n_jobs': -1, 'rf__n_estimators': 200, 'rf__min_samples_split': 3, 'rf__min_samples_leaf': 3, 'rf__max_depth': 12}
'''
