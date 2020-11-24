#Day12
#2020-11-24

# 리그레서 모델들 추출 
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
        kfold = KFold(n_splits=5, shuffle=True)
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name,' : ',scores)
        # model.fit(x_train,y_train)
        # y_pred =  model.predict(x_test)
        # print(name, '의 정답률 : ', r2_score(y_test,y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

'''
ARDRegression  :  [0.63472339 0.73699293 0.72638401 0.65501959 0.69614065]
AdaBoostRegressor  :  [0.81182737 0.83379332 0.84245069 0.77756577 0.85166409]
BaggingRegressor  :  [0.79138811 0.88575358 0.82851186 0.73332175 0.87658863]
BayesianRidge  :  [0.69691645 0.71099764 0.67201277 0.70536539 0.68568366]
CCA  :  [0.5806552  0.48904534 0.70327595 0.62285991 0.79428391]
DecisionTreeRegressor  :  [0.80223336 0.7540664  0.5059052  0.4440844  0.65935755]
DummyRegressor  :  [-0.00137973 -0.00129879 -0.04407133 -0.00123083 -0.0159433 ]
ElasticNet  :  [0.59493706 0.56272884 0.73365092 0.69541029 0.69606088]
ElasticNetCV  :  [0.60860904 0.67915763 0.57426626 0.62888366 0.73562787]
ExtraTreeRegressor  :  [0.86450746 0.74297366 0.53432261 0.59945686 0.64678139]
ExtraTreesRegressor  :  [0.85225421 0.90500975 0.86617287 0.85753211 0.87393061]
GammaRegressor  :  [-0.02701651 -0.00030033 -0.08118306 -0.00745336 -0.01016455]
GaussianProcessRegressor  :  [-6.78859393 -6.07186421 -7.37628514 -4.80213878 -6.2481255 ]    
GeneralizedLinearRegressor  :  [0.54524608 0.7250886  0.62371454 0.68799271 0.61748805]       
GradientBoostingRegressor  :  [0.92940103 0.7175701  0.87544309 0.7556643  0.87608506]        
HistGradientBoostingRegressor  :  [0.87467476 0.87732726 0.75257654 0.7248113  0.89037996]    
HuberRegressor  :  [0.69471431 0.67857872 0.70739335 0.65091386 0.47298981]
IsotonicRegression  :  [nan nan nan nan nan]
KNeighborsRegressor  :  [0.5832549  0.2546547  0.39651904 0.52064014 0.5169459 ]
KernelRidge  :  [0.43205815 0.73256777 0.78110944 0.64994815 0.58164087]
Lars  :  [0.79477887 0.75460525 0.58492768 0.76527814 0.60013652]
LarsCV  :  [0.49319365 0.81419339 0.79302459 0.53879881 0.72506674]
Lasso  :  [0.73905339 0.70775097 0.64044895 0.59981717 0.50907817]
LassoCV  :  [0.74348116 0.67173894 0.67785273 0.51023048 0.58180958]
LassoLars  :  [-0.00796666 -0.00699375 -0.00058403 -0.00392075 -0.00127822]
LassoLarsCV  :  [0.56913906 0.77362298 0.70654728 0.76041944 0.73332494]
LassoLarsIC  :  [0.7244167  0.7158275  0.69406338 0.75834145 0.57829223]
LinearRegression  :  [0.771467   0.64618242 0.70935878 0.64652762 0.69451981]
LinearSVR  :  [0.49898865 0.41101108 0.57294219 0.60247352 0.7069553 ]
MLPRegressor  :  [ 0.66048977 -0.35963132  0.47638036  0.49526194  0.29457928]
MultiTaskElasticNet  :  [nan nan nan nan nan]
MultiTaskElasticNetCV  :  [nan nan nan nan nan]
MultiTaskLasso  :  [nan nan nan nan nan]
MultiTaskLassoCV  :  [nan nan nan nan nan]
NuSVR  :  [0.1371885  0.22346531 0.21508614 0.09388968 0.25175281]
OrthogonalMatchingPursuit  :  [0.56054326 0.55417602 0.52423729 0.53617177 0.46495364]        
OrthogonalMatchingPursuitCV  :  [0.65705797 0.64562383 0.78680437 0.55099728 0.55431573]      
PLSCanonical  :  [-2.47160649 -0.79342764 -2.41959199 -1.97057166 -2.41194385]
PLSRegression  :  [0.66585045 0.59597438 0.68465931 0.73627246 0.70289204]
PassiveAggressiveRegressor  :  [ 0.12602799 -0.02234138  0.17265532  0.18889428 -0.08999591]  
PoissonRegressor  :  [0.7531553  0.70513508 0.8252041  0.73214539 0.68064051]
RANSACRegressor  :  [0.55351915 0.53909779 0.57947924 0.60876912 0.75013641]
RandomForestRegressor  :  [0.80693943 0.91920546 0.65201945 0.90496118 0.89685453]
Ridge  :  [0.66198069 0.6906941  0.65923349 0.7292856  0.72547757]
RidgeCV  :  [0.69031587 0.73322112 0.73959015 0.62832114 0.65868779]
SGDRegressor  :  [-1.74841079e+24 -4.33056960e+26 -6.08606957e+26 -2.10602576e+27
 -1.83006918e+26]
SVR  :  [0.1462815  0.18366299 0.03295133 0.21950759 0.36641889]
TheilSenRegressor  :  [0.61424655 0.60865322 0.75293689 0.73932156 0.619508  ]
TransformedTargetRegressor  :  [0.73375329 0.67349494 0.7734167  0.56167802 0.702008  ]       
TweedieRegressor  :  [0.60150164 0.69212087 0.67875117 0.67929497 0.56801638]
_SigmoidCalibration  :  [nan nan nan nan nan]
'''