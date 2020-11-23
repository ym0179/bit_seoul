#Day11
#2020-11-23

# 리그레서 모델들 추출 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)

x = iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=44)
allAlgorithms = all_estimators(type_filter='regressor') #리그레서 모든 모델들을 추출

for (name, algorithm) in allAlgorithms: #모든 모델들의 알고리즘
    model = algorithm()

    model.fit(x_train,y_train)
    y_pred =  model.predict(x_test)
    print(name, '의 정답률 : ', r2_score(y_test,y_pred))

import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

'''
pip uninstall scikit-learn
pip install scikit-learn == 0.20.1
=> 이제 안됨
'''

'''
ARDRegression 의 정답률 :  0.1766291380302032       
AdaBoostRegressor 의 정답률 :  0.26199764972842643  
BaggingRegressor 의 정답률 :  0.3352888131042977    
BayesianRidge 의 정답률 :  0.22292300805472887      
CCA 의 정답률 :  -0.11023563897039867
DecisionTreeRegressor 의 정답률 :  0.29123165515234417
DummyRegressor 의 정답률 :  -0.0005311221798414145  
ElasticNet 의 정답률 :  0.21764028536716706
ElasticNetCV 의 정답률 :  0.21685708793713876       
ExtraTreeRegressor 의 정답률 :  0.3879849884419534  
ExtraTreesRegressor 의 정답률 :  0.3640861661675838 
GammaRegressor 의 정답률 :  0.23982676648720092     
GaussianProcessRegressor 의 정답률 :  -1.7730259726513067
GeneralizedLinearRegressor 의 정답률 :  0.23920424928170336
GradientBoostingRegressor 의 정답률 :  0.35860791887281096
HistGradientBoostingRegressor 의 정답률 :  0.34338922721553367
HuberRegressor 의 정답률 :  0.20197825807917114
...error 더 이상 지원 X
'''