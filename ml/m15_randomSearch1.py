#Day12
#2020-11-24

# 랜덤 서치 random search
# 그리드 서치는 말그대로 모든 경우를 테이블로 만든뒤 격자로 탐색하는 방식에 해당한다면,
# 랜덤 서치는 하이퍼 파라미터 값을 랜덤하게 넣어보고 그중 우수한 값을 보인 하이퍼 파라미터를 활용해 모델을 생성
# 그리드 서치는 우리가 딕셔너리에 지정한 모든 값을 다 탐색해야만 함
# 불필요한 탐색 횟수를 줄임
# random Search는 중요한 hyper-parameter를 더 많이 탐색

# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# random search가 grid search의 50%인듯??

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,-1]
# print(x.shape, y.shape) #(150, 4) (150,)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

# SVM의 경우 
# 커널의 폭에 해당하는 gamma와 규제 매개변수 C가 중요
params = [
    {'C': [1,10,100,1000], 'kernel':['linear']},
    {'C': [1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C': [1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}
]

model = RandomizedSearchCV(SVC(), params, cv=kfold)


# 3. 훈련
model.fit(x_train,y_train) #model: GridSearch


# 4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_predict = model.predict(x_test)
print("최종정답률 : ", accuracy_score(y_test,y_predict))

# 최적의 매개변수 :  SVC(C=1000, gamma=0.0001)
# 최종정답률 :  1.0

'''
*SVM의 커널의 폭에 해당하는 gamma와 규제 매개변수 C

1) C는 얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는지를 결정
- 작을 수록 많이 허용하고, 클 수록 적게 허용 
- C값을 낮게 설정하면 이상치들이 있을 가능성을 크게 잡아 일반적인 결정 경계를 찾아냄
- 높게 설정하면 반대로 이상치의 존재 가능성을 작게 봐서 좀 더 세심하게 결정 경계를 찾아냄
- C가 너무 낮으면 과소적합(underfitting)이 될 가능성이 커짐
- C가 너무 높으면 과대적합(overfitting)이 될 가능성이 커짐

2) gamma는 하나의 데이터 샘플이 영향력을 행사하는 거리를 결정
- gamma는 가우시안 함수의 표준편차와 관련되어 있는데, 클수록 작은 표준편차를 가짐
- gamma가 클수록 한 데이터 포인터들이 영향력을 행사하는 거리가 짧아지는 반면
- gamma가 낮을수록 커진다
- gamma 매개변수는 결정 경계의 곡률을 조정한다고 말할 수도 있음
- 결정 경계가 결정 경계 가까이에 있는 데이터 샘플들에 영향을 크게 받기 때문에 점점 더 구불구불해짐

=> C는 데이터 샘플들이 다른 클래스에 놓이는 것을 허용하는 정도를 결정하고, gamma는 결정 경계의 곡률을 결정
'''