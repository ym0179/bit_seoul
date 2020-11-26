import pandas as pd
import numpy as np

#데이터 불러오기
train_identity = pd.read_csv("./data/project1/train_identity.csv")
train_transaction = pd.read_csv("./data/project1/train_transaction.csv")
test_identity = pd.read_csv("./data/project1/test_identity.csv")
test_transaction = pd.read_csv("./data/project1/test_transaction.csv")

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
# print(train.shape) # (590540, 434)
# print(test.shape) # (506691, 433)

# y값 분리
train_y = train['isFraud']
train_x = train.drop('isFraud', axis=1)
# print(train_x.shape) #(590540, 433)
# print(train_y.shape) #(590540,)

# 전처리 위해 train, test 값 합쳐줌
alldata = pd.concat([train_x,test])
# print(alldata)

alldata2 = pd.get_dummies(alldata)
alldata2 = alldata2.fillna(-1) #handle missing values
