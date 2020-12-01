# 모델링 파트
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import lightgbm
pd.options.display.max_colwidth=999
pd.options.display.max_rows=999
SEED = 77

#load data
x_train = np.load('./data/project1/x_train.npy',allow_pickle=True)
y_train = np.load('./data/project1/y_train.npy',allow_pickle=True)
test = np.load('./data/project1/x_test.npy',allow_pickle=True)
index = np.load('./data/project1/index_no_s.npy',allow_pickle=True)


#모델 테스트를 위해 부분 데이터 잘라서 사용 (데이터 양이 너무 많음) - random으로 40%
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=SEED, stratify=y_train)
x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, train_size=0.4, random_state=SEED, stratify=y_train)
x_test, x_temp, y_test, y_temp = train_test_split(x_test, y_test, train_size=0.4, random_state=SEED, stratify=y_test)
'''
#PCA 적용 해보기
pca = PCA(n_components=0.99) #데이터셋에 분산의 n%만 유지하도록 PCA를 적용
x_train_95 = pca.fit_transform(x_train)
x_test_95 = pca.transform(x_test)
print('원래 차원(픽셀) 수 :',  x_train.shape[1])
print('선택한 차원(픽셀) 수(0.99) :', pca.n_components_)
'''
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=SEED, stratify=y_train)
# x_train, x_val, y_train, y_val = train_test_split(x_train_95, y_train, train_size=0.8, random_state=SEED, stratify=y_train)

#random 40% 데이터
print("x train shape : ", x_train.shape) 
print("y train shape : ", y_train.shape)
print("x test shape : ", x_test.shape)
print("y test shape : ", y_test.shape)


params = {
    "n_estimators":[500, 800, 1000], #반복 수행하는 트리의 개수
    "learning_rate":[0.01, 0.05, 0.001],
    "num_leaves":[50], #하나의 트리가 가질 수 있는 최대 리프의 개수
    "max_depth ":[6, 10, 15, 20], # 트리의 최대 깊이,
    'min_child_samples':[20, 40, 60], #Leaf Node가 되기 위해서 최소한으로 필요한 데이터 개체의 수
    'subsample' :  [0.8,0.9], #데이터를 샘플링하는 비율
    'colsample_bytree ' : [0.5,0.7,1], #개별 트리를 학습할 때마다 무작위로 선택하는 피쳐의 비율을 제어
    'eval_metric' : ['auc']
    }
scoring = {
    'AUC': 'roc_auc',
}

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
lgbm = lightgbm.LGBMClassifier(
                            tree_method='gpu_hist', 
                            predictor='gpu_predictor',
                            random_state=SEED
                            )

model = RandomizedSearchCV(lgbm, params, n_jobs=-1, cv=kfold, scoring=scoring, n_iter=10, verbose=1, refit='AUC', return_train_score=True, random_state=SEED)

model.fit(x_train,y_train,
        eval_set=[(x_val, y_val)], 
        eval_metric="auc",
        early_stopping_rounds=100,
        )

df = pd.DataFrame(model.cv_results_)
print("cv result : \n", df.iloc[:,[18,12]])
df.T.to_csv('./data/project1/cv_results_.csv')

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 AUC : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

result = model.predict(x_test)
acc = accuracy_score(y_test,result)
print("acc : ", acc)

result2 = model.predict_proba(x_test)[:,1]
roc = roc_auc_score(y_test, result2)
print("AUC : %.4f%%"%(roc*100))
#roc curve 그리기
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, result2)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("roc_auc :", roc_auc)

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# top 20 feature importance by feature importance 값
def plot_feature_importances(model):
    plt.figure(figsize=(10,10))
    plt.title('Model Feature Importances')
    feature_names = index
    sorted_idx = model.feature_importances_.argsort()[::-1]
    plt.barh(feature_names[sorted_idx][:20][::-1], model.feature_importances_[sorted_idx][:20][::-1], align='center')
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)

plot_feature_importances(model)
plt.show()

