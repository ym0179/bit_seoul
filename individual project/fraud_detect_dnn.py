# 모델링 파트
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import lightgbm
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=SEED, stratify=y_train)

#random 40% 데이터
print("x train shape : ", x_train.shape) #(132280, 367)
print("y train shape : ", y_train.shape) #(132280,)
print("x test shape : ", x_test.shape) #(70864, 367)
print("y test shape : ", y_test.shape) #(70864,)


#모델링
model = Sequential()
model.add(Dense(1024, activation='relu',input_shape=(367,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc",tf.keras.metrics.AUC()])

es = EarlyStopping(monitor='loss',patience=30,mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=32,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 


#4. 평가
loss,acc,auc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("acc : ",acc)
print("auc : ",auc)


#5. 예측
# result = model.predict(x_test)
# acc = accuracy_score(y_test,result)
# print("acc : ", acc)

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
