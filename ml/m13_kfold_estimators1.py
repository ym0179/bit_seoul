#Day12
#2020-11-24

# 클래스파이어 모델들 추출
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=44)
allAlgorithms = all_estimators(type_filter='classifier') #클래스파이어 모든 모델들을 추출

for (name, algorithm) in allAlgorithms: #모든 모델들의 알고리즘
    try:    
        kfold = KFold(n_splits=7, shuffle=True)
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name,' : ',scores, " / ", scores.mean())
        # model.fit(x_train,y_train)
        # y_pred =  model.predict(x_test)
        # print(name, '의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함


'''
AdaBoostClassifier  :  [0.95833333 0.95833333 0.91666667 0.95833333 0.875     ]
BaggingClassifier  :  [0.91666667 1.         0.95833333 0.95833333 1.        ]
BernoulliNB  :  [0.08333333 0.20833333 0.33333333 0.33333333 0.20833333]
CalibratedClassifierCV  :  [0.875      1.         0.79166667 0.95833333 0.83333333]
CategoricalNB  :  [1.         0.91666667 0.83333333 1.         0.95833333]
CheckingClassifier  :  [0. 0. 0. 0. 0.]
ComplementNB  :  [0.70833333 0.70833333 0.66666667 0.58333333 0.625     ]
DecisionTreeClassifier  :  [0.91666667 1.         0.91666667 0.95833333 0.875     ]
DummyClassifier  :  [0.20833333 0.20833333 0.29166667 0.20833333 0.20833333]
ExtraTreeClassifier  :  [0.95833333 0.83333333 0.875      1.         0.95833333]
ExtraTreesClassifier  :  [0.91666667 1.         0.95833333 0.95833333 0.91666667]
GaussianNB  :  [1.         1.         1.         0.875      0.95833333]
GaussianProcessClassifier  :  [0.83333333 0.95833333 0.91666667 1.         0.95833333]        
GradientBoostingClassifier  :  [0.95833333 1.         0.95833333 0.91666667 0.95833333]       
HistGradientBoostingClassifier  :  [1.         0.91666667 0.875      0.91666667 0.91666667]   
KNeighborsClassifier  :  [0.95833333 0.875      1.         0.95833333 0.875     ]
LabelPropagation  :  [0.91666667 1.         0.91666667 0.95833333 0.875     ]
LabelSpreading  :  [0.91666667 0.95833333 0.95833333 0.95833333 0.95833333]
LinearDiscriminantAnalysis  :  [1.         0.95833333 0.95833333 0.91666667 0.95833333]       
LinearSVC  :  [0.91666667 0.91666667 0.95833333 1.         0.95833333]
LogisticRegression  :  [1.         0.95833333 0.95833333 0.91666667 0.91666667]
LogisticRegressionCV  :  [0.95833333 0.95833333 0.91666667 1.         1.        ]
MLPClassifier  :  [1.         0.91666667 0.875      1.         0.95833333]
MultinomialNB  :  [0.79166667 0.875      0.70833333 0.79166667 0.625     ]
NearestCentroid  :  [1.         1.         0.83333333 0.875      0.91666667]
NuSVC  :  [1.         0.875      0.91666667 0.95833333 1.        ]
PassiveAggressiveClassifier  :  [0.875      0.83333333 0.79166667 0.58333333 0.79166667]      
Perceptron  :  [0.33333333 0.54166667 0.79166667 0.58333333 0.91666667]
QuadraticDiscriminantAnalysis  :  [1.         0.95833333 0.875      1.         0.95833333]    
RadiusNeighborsClassifier  :  [0.91666667 0.95833333 0.95833333 0.875      0.95833333]        
RandomForestClassifier  :  [0.95833333 0.91666667 0.95833333 0.91666667 1.        ]
RidgeClassifier  :  [0.83333333 0.83333333 0.91666667 0.79166667 0.875     ]
RidgeClassifierCV  :  [0.83333333 0.79166667 0.91666667 0.95833333 0.79166667]
SGDClassifier  :  [0.875      0.79166667 0.58333333 0.70833333 0.95833333]
SVC  :  [0.91666667 1.         0.95833333 0.91666667 0.91666667]
'''