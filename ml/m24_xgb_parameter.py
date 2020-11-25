#Day13
#2020-11-25

#과적합 방지
#1. 훈련데이터량을 늘린다.
#2. 피처수를 줄인다.
#3. regularization

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


x,y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

n_estimators = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1
max_depth = 5
n_jobs = -1

model = XGBRegressor(max_depth=max_depth,                   # 트리의 최대 깊이를 정의 / 루트에서 가장 긴 노드의 거리 (보통 3 ~ 10)
                     learning_rate=learning_rate,           # 학습을 진행할 때마다 적용하는 학습률 (Default = 0.1, 범위 0 ~ 1) - Weak learner가 순차적으로 오류 값을 보정해나갈 때 적용하는 계수
                                                            # 낮은 만큼 최소 오류 값을 찾아 예측성능이 높아지지만, 많은 수의 트리가 필요하고 시간이 많이 소요
                     n_estimators=n_estimators,             # 생성할 트리의 갯수 (Default 100) - 많을소록 성능은 좋아지지만 시간이 오래 걸림
                     n_jobs=n_jobs,                         # -1 => 모든 코어를 다 쓰겠다
                     colsample_bylevel=colsample_bylevel,   # 각각의 트리 depth 마다 사용할 칼럼(Feature)의 비율 (보통 0.6 ~ 0.9)
                     colsample_bytree=colsample_bytree      # 각각의 트리(스탭)마다 사용할 칼럼(Feature)의 비율 (보통 0.6 ~ 0.9)
                     )

# model = XGBRegressor(max_depth=4) #without parameter setting

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

'''
without parameter setting
acc :  0.8225876771587038

with parameter
acc :  0.8963659312372875
'''