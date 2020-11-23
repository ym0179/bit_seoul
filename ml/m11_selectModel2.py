#Day11
#2020-11-23

# 리그레서 모델들 추출 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)
x = boston.iloc[:,0:12]
y = boston.iloc[:,12]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=44)
allAlgorithms = all_estimators(type_filter='regressor') #리그레서 모든 모델들을 추출

for (name, algorithm) in allAlgorithms: #모든 모델들의 알고리즘
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        y_pred =  model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test,y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

'''
pip uninstall scikit-learn
pip install scikit-learn == 0.20.1
=> 이제 안됨
'''

'''
ARDRegression 의 정답률 :  0.7413660842741397
AdaBoostRegressor 의 정답률 :  0.8461836136060118
BaggingRegressor 의 정답률 :  0.875268999312199
BayesianRidge 의 정답률 :  0.7397243134288036
CCA 의 정답률 :  0.7145358120880194
DecisionTreeRegressor 의 정답률 :  0.845835977656642
DummyRegressor 의 정답률 :  -0.0007982049217318821
ElasticNet 의 정답률 :  0.6952835513419808
ElasticNetCV 의 정답률 :  0.6863712064842076
ExtraTreeRegressor 의 정답률 :  0.5984745848421889
ExtraTreesRegressor 의 정답률 :  0.897427091055315
GammaRegressor 의 정답률 :  -0.0007982049217318821
GaussianProcessRegressor 의 정답률 :  -5.586473869478007
GeneralizedLinearRegressor 의 정답률 :  0.6899090511022785
GradientBoostingRegressor 의 정답률 :  0.8991708849240548
HistGradientBoostingRegressor 의 정답률 :  0.8843141840898427
HuberRegressor 의 정답률 :  0.7650865977198575
KNeighborsRegressor 의 정답률 :  0.6550811467209019
KernelRidge 의 정답률 :  0.7635967086119403
Lars 의 정답률 :  0.7440140846099281
LarsCV 의 정답률 :  0.7499770153318335
Lasso 의 정답률 :  0.683233856987759
LassoCV 의 정답률 :  0.7121285098074346
LassoLars 의 정답률 :  -0.0007982049217318821
LassoLarsCV 의 정답률 :  0.7477692079348518
LassoLarsIC 의 정답률 :  0.74479154708417
LinearRegression 의 정답률 :  0.7444253077310314
LinearSVR 의 정답률 :  0.6382072204007303
MLPRegressor 의 정답률 :  0.43746149976917703
NuSVR 의 정답률 :  0.32492104048309933
OrthogonalMatchingPursuit 의 정답률 :  0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률 :  0.7377665753906504
PLSCanonical 의 정답률 :  -1.3005198325202088
PLSRegression 의 정답률 :  0.7600229995900802
PassiveAggressiveRegressor 의 정답률 :  0.13207967065307247
PoissonRegressor 의 정답률 :  0.79037942981536
RANSACRegressor 의 정답률 :  0.6066115882069811
RandomForestRegressor 의 정답률 :  0.8857637770841255
Ridge 의 정답률 :  0.7465337048988421
RidgeCV 의 정답률 :  0.7452747021926976
SGDRegressor 의 정답률 :  -7.57587419191686e+26
SVR 의 정답률 :  0.2867592174963418
TheilSenRegressor 의 정답률 :  0.777314526843946
TransformedTargetRegressor 의 정답률 :  0.7444253077310314
TweedieRegressor 의 정답률 :  0.6899090511022785
'''