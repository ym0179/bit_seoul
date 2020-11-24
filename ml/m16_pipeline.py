#Day12
#2020-11-24

# 파이프라인을 사용하면 데이터 사전 처리 및 분류의 모든 단계를 포함하는 단일 개체를 만들 수 있다.
# - cross validation 할 때 train 만 fit+transform하고 val는 transform만 해야된다 => 과적합 방지
# - 파이프라인을 사용하면 가능
# 학습할 때와 동일한 기반 설정으로 동일하게 테스트 데이터를 변환해야 함 (train dataset 만 fit) - 훈련 데이터의 분포 추정 : 훈련 데이터를 입력으로 하여 fit 메서드를 실행하여 분포 모수를 객체내에 저장
# 학습 데이터에서 Scale된 데이터를 기반으로 모델이 학습이 되었기 때문에 
# 이렇게 학습된 모델이 예측을 할 때에도 학습 데이터의 Scale 기준으로 테스트/검증 데이터를 변환 한 뒤 predict

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,-1]
# print(x.shape, y.shape) #(150, 4) (150,)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler", MinMaxScaler()),('svm', SVC())])
pipe.fit(x_train,y_train)

print('acc : ', pipe.score(x_test,y_test)) #acc :  1.0

