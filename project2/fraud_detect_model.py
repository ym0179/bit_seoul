# 모델링 파트

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

#load data
x_train = np.load('./data/project1/x_train.npy',allow_pickle=True)
y_train = np.load('./data/project1/y_train.npy',allow_pickle=True)
test = np.load('./data/project1/x_test.npy',allow_pickle=True)
#파이썬에서 피클을 사용해 객체 배열(numpy 배열)을 저장할 수 있음 -> 배열의 내용이 일반 숫자 유형이 아닌 경우 (int/float) pickle를 사용해 array 저장

#shape
# print("x train shape : ", x_train.shape) #(590540, 367)
# print("y train shape : ", y_train.shape) #(590540,)
# print("x test shape : ", test.shape) #(506691, 367)

#모델 테스트를 위해 부분 데이터 잘라서 사용 (데이터 양이 너무 많음) - random으로 20%
x_train, x_temp, y_train, x_temp = train_test_split(x_train, y_train, train_size=0.01, random_state=77)
test, test_temp = train_test_split(test, train_size=0.01, random_state=77)

#random 20% 데이터
print("x train shape : ", x_train.shape) #(118108, 367)
print("y train shape : ", y_train.shape) #(118108,)
print("x test shape : ", test.shape) #(101338, 367)


params = {
    "n_estimators":[500, 800, 1000], 
    "learning_rate":[0.01,0.001], 
    "max_depth":range(3,10,3), 
    "colsample_bytree":[0.5,0.6,0.7], 
    "colsample_bylevel":[0.6,0.7,0.9],
    'min_child_weight':range(1,6,2),
    # 'subsample' :  [0.8] ,
    'objective' : ['binary:logistic'],
    'eval_metric' : ['auc'],
    'tree_method' : ['gpu_hist']
    }
# learning_rate default = 0.1
# colsample_bytree default = 1 (항상 모든 나무에서 중요한 칼럼에만 몰두해서 학습 -> 과적합 위험) / 학습할 칼럼 수가 많기 때문에 0.5-0.7까지 잡음
# max_depth default = 6 
# n_estimators default = 100 (learning rate를 낮게 잡아줬으니까 충분한 학습을 위해 늘려줌)
n_jobs = -1
# scoring = {
#     'AUC': 'roc_auc',
#     "Accuracy": make_scorer(accuracy_score)
# }

model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=n_jobs, cv=5, verbose=1)
# model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=n_jobs, cv=5, verbose=1, scoring=scoring, refit="AUC")
model.fit(x_train,y_train)

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

result = model.predict(test)
sc = model.score(test,result)
print("score : ", sc)

result2 = model.predict(test)[:,1]
print('ROC accuracy: {}'.format(roc_auc_score(test, val)))

import matplotlib.pyplot as plt
def plot_feature_importances(model):
    # n_features = x_train.shape[1]
    n_features = 10
    plt.barh(np.arange(n_features),np.sort(model.feature_importances_)[10], align='center')
    plt.yticks(np.arange(n_features), x_train.feature_names[10])
    plt.xlabel("Feature Imortances", size=15)
    plt.ylabel("Feautres", size=15)
    plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()


thresholds = np.sort(model.feature_importances_)
# print(thresholds)

save_score = 0
best_thresh = 0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    
    selection_model =  XGBClassifier(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)
    
    select_test = selection.transform(test)
    y_predict = selection_model.predict(select_test)

    score =  model.score(test,y_predict)

    # print("Thresh=%.4f, n=%d, acc: %.4f%%" %(thresh, select_x_train.shape[1], score))

    if score > save_score:
        save_score = score
        best_thresh = thresh
    # print("best_thresh, save_score: ", best_thresh, save_score)

print("best_thresh, save_score: ", best_thresh, save_score)

selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
x_train = selection.transform(x_train)
test = selection.transform(test)

model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=n_jobs, cv=5)

model.fit(x_train,y_train)

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

model = model.best_estimator_

result = model.predict(test)
acc = model.score(test,result)
print("acc : ", acc)
