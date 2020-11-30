# 모델링 파트
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import lightgbm
pd.options.display.max_colwidth=999
pd.options.display.max_rows=999
SEED = 77

#load data
x_train = np.load('./data/project1/x_train.npy',allow_pickle=True)
y_train = np.load('./data/project1/y_train.npy',allow_pickle=True)
test = np.load('./data/project1/x_test.npy',allow_pickle=True)
index = np.load('./data/project1/index_no_s.npy',allow_pickle=True)
#파이썬에서 피클을 사용해 객체 배열(numpy 배열)을 저장할 수 있음 -> 배열의 내용이 일반 숫자 유형이 아닌 경우 (int/float) pickle를 사용해 array 저장
# print("index :",index)

#shape
# print("x train shape : ", x_train.shape) #(590540, 367)
# print("y train shape : ", y_train.shape) #(590540,)
# print("x test shape : ", test.shape) #(506691, 367)

#모델 테스트를 위해 부분 데이터 잘라서 사용 (데이터 양이 너무 많음) - random으로 20%
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=SEED, stratify=y_train)
x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, train_size=0.4, random_state=SEED, stratify=y_train)
x_test, x_temp, y_test, y_temp = train_test_split(x_test, y_test, train_size=0.4, random_state=SEED, stratify=y_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=SEED, stratify=y_train)

#random 20% 데이터
print("x train shape : ", x_train.shape) #(118108, 367)
print("y train shape : ", y_train.shape) #(118108,)
print("x test shape : ", x_test.shape) #(101338, 367)
print("y test shape : ", y_test.shape) #(101338, 367)


params = {
    "n_estimators":[500, 800, 1000], #  default=100, 반복 수행하는 트리의 개수 - 너무 크게 지정하면 학습 시간이 오래 걸리고 과적합이 발생
    "learning_rate":[0.01, 0.05, 0.001], # learning_rate default = 0.1, n_estimators와 같이 튜닝
    "num_leaves":[50], #하나의 트리가 가질 수 있는 최대 리프의 개수 - 높이면 정확도는 높아지지만 트리의 깊이가 커져 모델의 복잡도가 증가
    "max_depth ":[6, 10, 15, 20], # default=3, 트리의 최대 깊이, 과적합을 방지하기 위해 num_leaves는 2^(max_depth)보다 작아야함
    'min_child_samples':[20, 40, 60], #최종 결정 클래스인 Leaf Node가 되기 위해서 최소한으로 필요한 데이터 개체의 수, 과적합 제어
    #너무 큰 숫자로 설정하면 under-fitting이 일어날 수 있으며, 아주 큰 데이터셋이라면 적어도 수백~수천 정도로 가정하는 것이 편리
    'subsample' :  [0.8,0.9], #과적합을 제어하기 위해 데이터를 샘플링하는 비율
    'colsample_bytree ' : [0.5,0.7,1], #default=1.0, 개별 트리를 학습할 때마다 무작위로 선택하는 피쳐의 비율을 제어
    # 'objective' : ['binary:logistic'],
    # 'early_stopping_rounds' : [10]
    'eval_metric' : ['auc']
    # 'tree_method' : ['gpu_hist']
    }

scoring = {
    'AUC': 'roc_auc',
}

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lgbm = lightgbm.LGBMClassifier(
                            tree_method='gpu_hist', 
                            predictor='gpu_predictor',
                            random_state=SEED
                            )

model = RandomizedSearchCV(lgbm, params, n_jobs=-1, cv=kfold, scoring=scoring, n_iter=10, verbose=1, refit='AUC', return_train_score=True, random_state=SEED)
# Scoring: 평가 기준으로 할 함수 / cv: int, 교차검증 생성자 또는 반복자 / n_iter: int, 몇 번 반복하여 수행할 것인지에 대한 값
# model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=n_jobs, cv=5, verbose=1, scoring=scoring, refit="AUC")

model.fit(x_train,y_train,
        eval_set=[(x_val, y_val)], 
        eval_metric="auc",
        early_stopping_rounds=100,
        )

df = pd.DataFrame(model.cv_results_)
# print("cv result : \n", df.loc[:,['mean_test_AUC', 'params']])
print("cv result : \n", df.iloc[:,[18,12]])

df.T.to_csv('./data/project1/cv_results_.csv')

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 AUC : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

result = model.predict(x_test)
sc = model.score(x_test,y_test)
print("score : ", sc)

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

# top 20 feature importance by plot_importance
# plot_importance(model, max_num_features=20)
# plt.show()

# top 20 feature importance by feature importance 값
def plot_feature_importances(model):
    # n_features = x_train.shape[1]
    # n_features = 10
    plt.figure(figsize=(10,10))
    plt.title('Model Feature Importances')
    feature_names = index
    sorted_idx = model.feature_importances_.argsort()[::-1]
    # print("sorted_idx: ",sorted_idx)
    plt.barh(feature_names[sorted_idx][:20][::-1], model.feature_importances_[sorted_idx][:20][::-1], align='center')
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)
    # plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()


thresholds = np.sort(model.feature_importances_)
print(thresholds)


save_score = 0
best_thresh = 0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_test = selection.transform(x_test)

    selection_model = LGBMClassifier(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict_proba(select_test)[:,1]
    # score =  model.score(test,y_predict)
    roc = roc_auc_score(y_test, result2)

    print("Thresh=%.4f, n=%d, roc: %.4f%%" %(thresh, select_x_train.shape[1], roc))

    if roc > save_score:
        save_score = roc
        best_thresh = thresh
    print("best_thresh, save_score: ", best_thresh, save_score)

print("=======================================")
print("best_thresh, save_score: ", best_thresh, save_score)

selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
x_train = selection.transform(x_train)
x_test = selection.transform(x_test)

# model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=-1, cv=5)
model = RandomizedSearchCV(lgbm, params, n_jobs=-1, cv=kfold, verbose=1, scoring=scoring, n_iter=5, refit='AUC', return_train_score=True, random_state=SEED)

model.fit(x_train,y_train)

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 AUC : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

result = model.predict(x_test)
sc = model.score(x_test,y_test)
print("score : ", sc)

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
plt.show()