# 모델링 파트
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import xgboost
pd.options.display.max_colwidth=999
pd.options.display.max_rows=999
SEED = 77


#load data
x_train = np.load('./data/project1/x_train.npy',allow_pickle=True)
y_train = np.load('./data/project1/y_train.npy',allow_pickle=True)
test = np.load('./data/project1/x_test.npy',allow_pickle=True)
index = np.load('./data/project1/index_no_s.npy',allow_pickle=True)


#모델 테스트를 위해 부분 데이터 잘라서 사용 (데이터 양이 너무 많음) - random으로 40%
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=SEED,stratify=y_train)
x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, train_size=0.4, random_state=SEED,stratify=y_train)
x_test, x_temp, y_test, y_temp = train_test_split(x_test, y_test, train_size=0.4, random_state=SEED,stratify=y_test)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=SEED, stratify=y_train)


#random 20% 데이터
print("x train shape : ", x_train.shape)
print("y train shape : ", y_train.shape)
print("x test shape : ", x_test.shape)
print("y test shape : ", y_test.shape)

params = {
    "n_estimators":[500, 800, 1000, 1200], 
    "learning_rate":[0.01, 0.05, 0.001], 
    "max_depth":range(3,10,3),
    "colsample_bytree":[0.5,0.6,0.7],    
    "colsample_bylevel":[0.7,0.8,0.9],
    'min_child_weight':range(1,6,2),
    'subsample' :  [0.8,0.9],
    'objective' : ['binary:logistic'],
    }
scoring = {
    'AUC': 'roc_auc',
}

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
xgb = xgboost.XGBClassifier(tree_method='gpu_hist', 
                            predictor='gpu_predictor',
                            reg_alpha=0.15, #default 1, 능선 회쉬(Ridge Regression)의 L2 정규화
                            reg_lamdba=0.85, #default 0, 라쏘 회귀(Lasso Regression)의 L1 정규화
                            random_state=SEED
                            )

model = RandomizedSearchCV(xgb, params, n_jobs=-1, cv=kfold, scoring=scoring, 
                        n_iter=5, verbose=1, refit='AUC', return_train_score=True, random_state=SEED)

model.fit(x_train,y_train,
        eval_set=[(x_val, y_val)], 
        eval_metric="auc",
        early_stopping_rounds=100,
        )

#cv 별 결과 출력
df = pd.DataFrame(model.cv_results_)
print("cv result : \n", df.iloc[:,[18,12]])
df.T.to_csv('./data/project1/cv_results_.csv')

print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 AUC : {0:.4f}".format(model.best_score_))
model = model.best_estimator_

result = model.predict(x_test)
acc = accuracy_score(y_test,result)
print("Accuracy : ", acc)

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


thresholds = np.sort(model.feature_importances_)
save_score = 0
best_thresh = 0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_test = selection.transform(x_test)

    selection_model = XGBClassifier(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict_proba(select_test)[:,1]
    roc = roc_auc_score(y_test, result2)

    # print("Thresh=%.4f, n=%d, roc: %.4f%%" %(thresh, select_x_train.shape[1], roc))

    if roc > save_score:
        save_score = roc
        best_thresh = thresh
    # print("best_thresh, save_score: ", best_thresh, save_score)

# print("=======================================")
# print("best_thresh, save_score: ", best_thresh, save_score)

selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
x_train = selection.transform(x_train)
x_test = selection.transform(x_test)

model = RandomizedSearchCV(xgb, params, n_jobs=-1, cv=kfold, verbose=1, scoring=scoring, n_iter=3, refit='AUC', return_train_score=True, random_state=SEED)
model.fit(x_train,y_train)

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
plt.show()