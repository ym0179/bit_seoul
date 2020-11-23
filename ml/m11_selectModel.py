#Day11
#2020-11-23

# 클래스파이어 모델들 추출
import pandas as pd
from sklearn.model_selection import train_test_split
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
    model = algorithm()

    model.fit(x_train,y_train)
    y_pred =  model.predict(x_test)
    print(name, '의 정답률 : ', accuracy_score(y_test,y_pred))

import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

'''
pip uninstall scikit-learn
pip install scikit-learn == 0.20.1
=> 이제 안됨
'''

'''
AdaBoostClassifier 의 정답률 :  0.9666666666666667  
BaggingClassifier 의 정답률 :  0.9666666666666667   
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9333333333333333
CategoricalNB 의 정답률 :  0.9
CheckingClassifier 의 정답률 :  0.3
...error 더 이상 지원 X
'''