#Day10
#2020-11-20

import pandas as pd
import numpy as np

####################### 데이터 불러오기 ################################
bit = pd.read_csv('./data/csv/비트컴퓨터 1120.csv', header=0, index_col=None, sep=',', encoding='CP949')
samsung = pd.read_csv('./data/csv/삼성전자 1120.csv', header=0, index_col=None, sep=',', encoding='CP949')
gold = pd.read_csv('./data/csv/금현물.csv', header=0, index_col=None, sep=',', encoding='CP949')
kosdaq = pd.read_csv('./data/csv/코스닥.csv', header=0, index_col=None, sep=',', encoding='CP949')

#Unnamed: 6 컬럼 이름 바꾸기
bit.rename(columns = {"Unnamed: 6": "전일대비"}, inplace = True)
samsung.rename(columns = {"Unnamed: 6": "전일대비"}, inplace = True)
gold.rename(columns = {"Unnamed: 6": "전일대비"}, inplace = True)

#컬럼 빼기
bit = bit.drop(['전일비','신용비','외인비','고가','저가','등락률','거래량','금액(백만)','전일대비'],axis=1) #8열
samsung = samsung.drop(['전일비','신용비','외인비','고가','저가','등락률'],axis=1) #11열
gold = gold.drop(['전일비','고가','저가','등락률'],axis=1) #9열
kosdaq = kosdaq.drop(['전일대비','Unnamed: 6','고가','저가','등락률','상승','상한','보합','하락','하한'],axis=1) #5열
# print(bit)
# print(samsung)
# print(gold)
# print(kosdaq)

#11/20일자 데이터 삭제
bit = bit.drop([bit.index[0]])
samsung = samsung.drop([samsung.index[0]])
gold = gold.drop([gold.index[0]])
kosdaq = kosdaq.drop([kosdaq.index[0]])


#정렬을 오름차순으로 변경
bit = bit.sort_values(['일자'], ascending=True)
samsung = samsung.sort_values(['일자'], ascending=True)
gold = gold.sort_values(['일자'], ascending=True)
kosdaq = kosdaq.sort_values(['일자'], ascending=True)

#reset index (오름차순 해줬으니까)
bit.reset_index(drop=True,inplace=True)
samsung.reset_index(drop=True,inplace=True)
gold.reset_index(drop=True,inplace=True)
kosdaq.reset_index(drop=True,inplace=True)

#일자 컬럼 빼기
bit = bit.drop('일자',axis=1)
samsung = samsung.drop('일자',axis=1)
gold = gold.drop('일자',axis=1)
kosdaq = kosdaq.drop('일자',axis=1)

# index 32번전까지 자르기 (삼성 액면분할)
samsung = samsung.drop(samsung.index[0:33])
samsung.reset_index(drop=True,inplace=True) #reset index

# print(samsung.shape) #(626, 10)
# print(bit.shape) #(1199, 7)
# print(gold.shape) #(809, 8)
# print(kosdaq.shape) #(879, 4)


#삼성이랑 비트/금/코스닥 행 개수 맞춰주기
bit = bit.drop(bit.index[0:573])
bit.reset_index(drop=True,inplace=True) #reset index
gold = gold.drop(gold.index[0:183])
gold.reset_index(drop=True,inplace=True) #reset index
kosdaq = kosdaq.drop(kosdaq.index[0:253])
kosdaq.reset_index(drop=True,inplace=True) #reset index

# print(samsung.shape) #(626, 10)
# print(bit.shape) #(626, 7)
# print(gold.shape) #(626, 8)
# print(kosdaq.shape) #(626, 4)

gold = gold.astype('str')

# 숫자에 콤마 제거 후 문자를 정수로 변환
for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',""))

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',""))

for i in range(len(gold.index)):
    for j in range(len(gold.iloc[i])):
        gold.iloc[i,j] = int(gold.iloc[i,j].replace(',',""))

for i in range(len(kosdaq.index)):
    for j in range(2,len(kosdaq.iloc[i])):
        kosdaq.iloc[i,j] = int(kosdaq.iloc[i,j].replace(',',""))


#numpy로 바꿈
bit_np = bit.to_numpy()
samsung_np = samsung.to_numpy()
gold_np = gold.to_numpy()
kosdaq_np = kosdaq.to_numpy()

# print(samsung.shape) #(626, 10)
# print(bit.shape) #(626, 7)
# print(gold.shape) #(626, 8)
# print(kosdaq.shape) #(626, 4)

# x, y 나누기
# y 값은 시가!! => 1번째 컬럼
samsung_y = samsung_np[:,0]
samsung_x = samsung_np[:,[1,2,3,4,5,6,7,8,9]] #y열 제거
bit_y = bit_np[:,0]
bit_x = bit_np[:,[1,2,3,4,5,6]] #y열 제거
gold_y = gold_np[:,0]
gold_x = gold_np[:,[1,2,3,4,5,6,7]] #y열 제거
kosdaq_y = kosdaq_np[:,0]
kosdaq_x = kosdaq_np[:,[1,2,3]] #y열 제거


#데이터 나누기 (시계열 데이터로 5행씩 나누기)
def split_x2(seq, size):
    aaa = []
    # print("size :",size)
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bit_x = split_x2(bit_x, 5)
samsung_x = split_x2(samsung_x, 5)
gold_x = split_x2(gold_x, 5)
kosdaq_x = split_x2(kosdaq_x, 5)

# print(samsung_x.shape) #(622, 5, 9)
# print(bit_x.shape) #(622, 5, 6)
# print(gold_x.shape) #(622, 5, 7)
# print(kosdaq_x.shape) #(622, 5, 3)


#x dataset은 마지막 데이터는 x_predict로 쓸꺼니까 따로 빼주고 x dataset에서 마지막 데이터(행)과 마지막에서 두번째 행 제거
bit_pred = bit_x[-1]
samsung_pred = samsung_x[-1]
gold_pred = gold_x[-1]
kosdaq_pred = kosdaq_x[-1]

# print(bit_x.shape) #(621, 5, 6)
# print(samsung_x.shape) #(621, 5, 9)
# print(gold_x.shape) #(621, 5, 7)
# print(kosdaq_x.shape) #(621, 5, 3)

bit_x=np.delete(bit_x,[621],axis=0) #마지막 행 제거
samsung_x=np.delete(samsung_x,[621],axis=0) #마지막 행 제거
gold_x=np.delete(gold_x,[621],axis=0) #마지막 행 제거
kosdaq_x=np.delete(kosdaq_x,[621],axis=0) #마지막 행 제거

bit_x=np.delete(bit_x,[620],axis=0) #마지막 행 제거
samsung_x=np.delete(samsung_x,[620],axis=0) #마지막 행 제거
gold_x=np.delete(gold_x,[620],axis=0) #마지막 행 제거
kosdaq_x=np.delete(kosdaq_x,[620],axis=0) #마지막 행 제거

# print(bit_x.shape) #(620, 5, 6)
# print(samsung_x.shape) #(620, 5, 9)
# print(gold_x.shape) #(620, 5, 7)
# print(kosdaq_x.shape) #(620, 5, 3)


#y dataset은 size=5니까 맨 앞에 6개 행 빼기
bit_y=np.delete(bit_y,[0,1,2,3,4,5],axis=0) 
samsung_y=np.delete(samsung_y,[0,1,2,3,4,5],axis=0) 
gold_y=np.delete(gold_y,[0,1,2,3,4,5],axis=0) 
kosdaq_y=np.delete(kosdaq_y,[0,1,2,3,4,5],axis=0) 

# print(bit_y.shape) #(620,)
# print(samsung_y.shape) #(620,)
# print(gold_y.shape) #(620,)
# print(kosdaq_y.shape) #(620,)

#npy 저장
np.save('./data/bit_x.npy', arr=bit_x)
np.save('./data/bit_y.npy', arr=bit_y)
np.save('./data/bit_pred.npy', arr=bit_pred)
np.save('./data/samsung_x.npy', arr=samsung_x)
np.save('./data/samsung_y.npy', arr=samsung_y)
np.save('./data/samsung_pred.npy', arr=samsung_pred)
np.save('./data/gold_x.npy', arr=gold_x)
np.save('./data/gold_y.npy', arr=gold_y)
np.save('./data/gold_pred.npy', arr=gold_pred)
np.save('./data/kosdaq_x.npy', arr=kosdaq_x)
np.save('./data/kosdaq_y.npy', arr=kosdaq_y)
np.save('./data/kosdaq_pred.npy', arr=kosdaq_pred)


####################### 1.데이터 전처리 ################################
#float type 변환
samsung_x = samsung_x.astype('float32')
samsung_y = samsung_y.astype('float32')
samsung_pred = samsung_pred.astype('float32')
bit_x = bit_x.astype('float32')
bit_y = bit_y.astype('float32')
bit_pred = bit_pred.astype('float32')
gold_x = gold_x.astype('float32')
gold_y = gold_y.astype('float32')
gold_pred = gold_pred.astype('float32')
kosdaq_x = kosdaq_x.astype('float32')
kosdaq_y = kosdaq_y.astype('float32')
kosdaq_pred = kosdaq_pred.astype('float32')

#train-test split
from sklearn.model_selection import train_test_split
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test, bit_y_train, bit_y_test = train_test_split(bit_x, bit_y, train_size=0.8)
gold_x_train, gold_x_test, gold_y_train, gold_y_test = train_test_split(gold_x, gold_y, train_size=0.8)
kosdaq_x_train, kosdaq_x_test, kosdaq_y_train, kosdaq_y_test = train_test_split(kosdaq_x, kosdaq_y, train_size=0.8)


samsung_pred = samsung_pred.reshape(1,samsung_pred.shape[0],samsung_pred.shape[1]) #pred 값 3차원으로 맞추기
bit_pred = bit_pred.reshape(1,bit_pred.shape[0],bit_pred.shape[1]) #pred 값 3차원으로 맞추기
gold_pred = gold_pred.reshape(1,gold_pred.shape[0],gold_pred.shape[1]) #pred 값 3차원으로 맞추기
kosdaq_pred = kosdaq_pred.reshape(1,kosdaq_pred.shape[0],kosdaq_pred.shape[1]) #pred 값 3차원으로 맞추기


# scaling for 삼성
from sklearn.preprocessing import StandardScaler
num_sample   = samsung_x_train.shape[0] # 샘플 데이터 수
num_sequence = samsung_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_x_train.shape[2] # Feature 수 (=9)
scaler = StandardScaler()
# 시계열을 선회하면서 피팅
for ss in range(num_sequence):
    scaler.partial_fit(samsung_x_train[:, ss, :]) #fit은 train data만 함
# 스케일링(변환)
# 1. samsung train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_train = np.concatenate(results, axis=1)
# 2. samsung test data 다시 선언 => sampe data 수가 (train과) 다르니까??
num_sample   = samsung_x_test.shape[0] # 샘플 데이터 수 
num_sequence = samsung_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_x_test.shape[2] # Feature 수 (=9)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_test = np.concatenate(results, axis=1)
# 3. samsung predict data
num_sample   = samsung_pred.shape[0] # 샘플 데이터 수
num_sequence = samsung_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_pred.shape[2] # Feature 수 (=9)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_pred = np.concatenate(results, axis=1)

# scaling for 비트
num_sample   = bit_x_train.shape[0] # 샘플 데이터 수
num_sequence = bit_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_x_train.shape[2] # Feature 수 (=6)
scaler = StandardScaler()
for ss in range(num_sequence):
    scaler.partial_fit(bit_x_train[:, ss, :]) #fit은 train data만 함
# 1. bit train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_x_train = np.concatenate(results, axis=1)
# 2. bit test data
num_sample   = bit_x_test.shape[0] # 샘플 데이터 수 
num_sequence = bit_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_x_test.shape[2] # Feature 수 (=6)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_x_test = np.concatenate(results, axis=1)
# 3. bit predict data
num_sample   = bit_pred.shape[0] # 샘플 데이터 수
num_sequence = bit_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_pred.shape[2] # Feature 수 (=6)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_pred = np.concatenate(results, axis=1)

# scaling for 금현물
num_sample   = gold_x_train.shape[0] # 샘플 데이터 수
num_sequence = gold_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = gold_x_train.shape[2] # Feature 수 (=7)
scaler = StandardScaler()
for ss in range(num_sequence):
    scaler.partial_fit(gold_x_train[:, ss, :]) #fit은 train data만 함
# 1. gold train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(gold_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
gold_x_train = np.concatenate(results, axis=1)
# 2. gold test data
num_sample   = gold_x_test.shape[0] # 샘플 데이터 수 
num_sequence = gold_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = gold_x_test.shape[2] # Feature 수 (=7)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(gold_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
gold_x_test = np.concatenate(results, axis=1)
# 3. gold predict data
num_sample   = gold_pred.shape[0] # 샘플 데이터 수
num_sequence = gold_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = gold_pred.shape[2] # Feature 수 (=7)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(gold_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
gold_pred = np.concatenate(results, axis=1)


# scaling for 코스닥
num_sample   = kosdaq_x_train.shape[0] # 샘플 데이터 수
num_sequence = kosdaq_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = kosdaq_x_train.shape[2] # Feature 수 (=3)
scaler = StandardScaler()
for ss in range(num_sequence):
    scaler.partial_fit(kosdaq_x_train[:, ss, :]) #fit은 train data만 함
# 1. kosdaq train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(kosdaq_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
kosdaq_x_train = np.concatenate(results, axis=1)
# 2. kosdaq test data
num_sample   = kosdaq_x_test.shape[0] # 샘플 데이터 수 
num_sequence = kosdaq_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = kosdaq_x_test.shape[2] # Feature 수 (=3)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(kosdaq_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
kosdaq_x_test = np.concatenate(results, axis=1)
# 3. kosdaq predict data
num_sample   = kosdaq_pred.shape[0] # 샘플 데이터 수
num_sequence = kosdaq_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = kosdaq_pred.shape[2] # Feature 수 (=3)
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(kosdaq_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
kosdaq_pred = np.concatenate(results, axis=1)

# print(bit_x.shape) #(620, 5, 6)
# print(samsung_x.shape) #(620, 5, 9)
# print(gold_x.shape) #(620, 5, 7)
# print(kosdaq_x.shape) #(620, 5, 3)
# print(samsung_y.shape) #(620, 5, 9)


####################### 2.모델링 => LSTM(2차원 인풋), Conv1D(2차원 인풋) ################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D

#모델 1.
input1 = Input(shape=(5,9)) #입력 1
layer1_1 = LSTM(64, activation='relu', return_sequences=True)(input1)
layer1_2 = LSTM(128, activation='relu')(layer1_1)
layer1_3 = Dense(128, activation='relu')(layer1_2)
layer1_4 = Dropout(0.3)(layer1_3)
layer1_5 = Dense(64, activation='relu')(layer1_4)
output1 = Dense(32, activation='relu')(layer1_5)


#모델 2.
input2 = Input(shape=(5,6)) #입력 2
layer2_1 = LSTM(64, activation='relu', return_sequences=True)(input1)
layer2_2 = LSTM(128, activation='relu')(layer2_1)
layer2_3 = Dense(128, activation='relu')(layer2_2)
layer2_4 = Dropout(0.3)(layer2_3)
layer2_5 = Dense(128, activation='relu')(layer2_4)
output2 = Dense(64, activation='relu')(layer2_5)

# layer2_1 = Conv1D(128, (2), padding="same")(input2)
# layer2_2 = MaxPooling1D(pool_size=2)(layer2_1)
# layer2_3 = Dropout(0.3)(layer2_2)
# layer2_4 = Conv1D(64, (2), padding="same")(layer2_3)
# layer2_5 = MaxPooling1D(pool_size=2)(layer2_4)
# layer2_6 = Dropout(0.3)(layer2_5)
# layer2_7 = Flatten()(layer2_6)
# output2 = Dense(256, activation='relu')(layer2_7)


#모델 3.
input3 = Input(shape=(5,7)) #입력 3
layer3_1 = LSTM(128, activation='relu')(input3)
layer3_2 = Dense(128, activation='relu')(layer3_1)
layer3_3 = Dropout(0.3)(layer3_2)
output3 = Dense(128, activation='relu')(layer3_3)

#모델 4.
input4 = Input(shape=(5,3)) #입력 4
layer4_1 = LSTM(128, activation='relu')(input4)
layer4_2 = Dense(128, activation='relu')(layer4_1)
layer4_3 = Dropout(0.3)(layer4_2)
layer4_4 = Dense(128, activation='relu')(layer4_3)
output4 = Dense(32, activation='relu')(layer4_4)

############### 모델 병합, concatenate ###################
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1,output2,output3,output4])
middle1 = Dense(128, activation='relu')(merge1)
middle1 = Dense(64, activation='relu')(middle1)

################ output 모델 구성 #####################
output = Dense(128, activation='relu')(middle1)
output = Dense(1)(output) #출력 1

#모델 정의
model = Model(inputs=[input1,input2,input3,input4],
              outputs=output)
# model.summary()


####################### #3. 컴파일, 훈련 ################################
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=100,mode='auto')
modelpath = './model/samsung2_2.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit([samsung_x_train,bit_x_train,gold_x_train,kosdaq_x_train],samsung_y_train,epochs=1000,batch_size=32,verbose=2,
        callbacks=[es,cp], validation_split=0.2) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/samsung2_2.hdf5')


#4. 평가
loss,mae = model.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],
                        samsung_y_test,
                        batch_size=32)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
result = model.predict([samsung_pred,bit_pred,gold_pred,kosdaq_pred])
print("삼성 시가 예측값 : ", result)


#R2
y_predicted = model.predict([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test])
from sklearn.metrics import r2_score
r2 = r2_score(samsung_y_test, y_predicted)
print("R2 : ",r2) # max 값: 1