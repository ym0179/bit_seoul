#Day10
#2020-11-20

import pandas as pd
import numpy as np

####################### 데이터 불러오기 ################################
bit = pd.read_csv('./data/csv/비트컴퓨터 1120.csv', header=0, index_col=None, sep=',', encoding='CP949')
samsung = pd.read_csv('./data/csv/삼성전자 1120.csv', header=0, index_col=None, sep=',', encoding='CP949')
# print(bit)
# print(samsung)

#Unnamed: 6 컬럼 이름 바꾸기
bit.rename(columns = {"Unnamed: 6": "전일대비"}, inplace = True)
samsung.rename(columns = {"Unnamed: 6": "전일대비"}, inplace = True)

#컬럼 빼기
bit = bit.drop(['전일비','신용비','외인비','고가','저가','등락률','거래량','금액(백만)'],axis=1)
samsung = samsung.drop(['전일비','신용비','외인비','고가','저가','등락률'],axis=1)

#11/20일자 데이터 삭제
bit = bit.drop([bit.index[0]])
samsung = samsung.drop([samsung.index[0]])

#정렬을 오름차순으로 변경
bit = bit.sort_values(['일자'], ascending=True)
samsung = samsung.sort_values(['일자'], ascending=True)

#reset index (오름차순 해줬으니까)
bit.reset_index(drop=True,inplace=True)
samsung.reset_index(drop=True,inplace=True)

#컬럼 빼기
bit = bit.drop('일자',axis=1)
samsung = samsung.drop('일자',axis=1)

# index 32번전까지 자르기 (삼성 액면분할)
samsung = samsung.drop(samsung.index[0:33])
samsung.reset_index(drop=True,inplace=True) #reset index

#삼성이랑 비트 행 개수 맞춰주기
bit = bit.drop(bit.index[0:573])
bit.reset_index(drop=True,inplace=True) #reset index

# 숫자에 콤마 제거 후 문자를 정수로 변환
for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',""))

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',""))

bit_np = bit.to_numpy()
samsung_np = samsung.to_numpy()

# x, y 나누기
# y 값은 종가!! => 2번째 컬럼
samsung_y = samsung_np[:,1]
samsung_x = samsung_np[:,[0,2,3,4,5,6,7,8,9]] #y열 제거
bit_y = bit_np[:,1]
bit_x = bit_np[:,[0,2,3,4,5,6,7]] #y열 제거
print(samsung_y.shape)
print(samsung_x)
print(samsung_x.shape)

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

#x dataset은 마지막 데이터는 x_predict로 쓸꺼니까 따로 빼주고 x dataset에서 마지막 데이터(행) 제거
bit_pred = bit_x[-1]
samsung_pred = samsung_x[-1]
# print(bit_x[-1])
# print(bit_pred)
# print(bit_x[-2])
bit_x=np.delete(bit_x,[621],axis=0) #마지막 행 제거
# print(bit_x[-1])
samsung_x=np.delete(samsung_x,[621],axis=0) #마지막 행 제거
print(bit_x.shape) #(621, 5, 7)
print(samsung_x.shape) #(621, 5, 9)
print(samsung_pred)
print(bit_pred)

#y dataset은 size=5니까 맨 앞에 5개 행 빼기
bit_y=np.delete(bit_y,[0,1,2,3,4],axis=0) 
samsung_y=np.delete(samsung_y,[0,1,2,3,4],axis=0) 
# print(bit_y.shape) #(621,)
# print(samsung_y.shape) #(621,)

#npy 저장
np.save('./data/bit_x.npy', arr=bit_x)
np.save('./data/samsung_x.npy', arr=samsung_x)
np.save('./data/bit_y.npy', arr=bit_y)
np.save('./data/samsung_y.npy', arr=samsung_y)

print("===============================")

####################### 1.데이터 전처리 ################################
#float type 변환
samsung_x = samsung_x.astype('float32')
samsung_y = samsung_y.astype('float32')
bit_x = bit_x.astype('float32')
bit_y = bit_y.astype('float32')
bit_pred = bit_pred.astype('float32')
samsung_pred = samsung_pred.astype('float32')

#train-test split
from sklearn.model_selection import train_test_split
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8)

bit_x_train, bit_x_test, bit_y_train, bit_y_test = train_test_split(bit_x, bit_y, train_size=0.8)

samsung_pred = samsung_pred.reshape(1,samsung_pred.shape[0],samsung_pred.shape[1]) #pred 값 3차원으로 맞추기
bit_pred = bit_pred.reshape(1,bit_pred.shape[0],bit_pred.shape[1]) #pred 값 3차원으로 맞추기

'''
# *시계열이 포함된 3차원 데이터 스케일링 하는 방법*
# 방법1) 2차원으로 reshape( )하고 스케일링 => 2차원으로 형상을 바꾸고 스케일링 했을 때 예기치 못한 부작용이 있을 수 있음
# => 3차원에 시계열이 포함된 경우 각 시간 슬라이스별 스케일이 다를 수 있기 때문에 (부작용)

# 방법2) 3차원 시계열 축을 순회하면서 스케일링 (추천)
'''
# scaling for 삼성
from sklearn.preprocessing import MinMaxScaler

num_sample   = samsung_x_train.shape[0] # 샘플 데이터 수
num_sequence = samsung_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_x_train.shape[2] # Feature 수 (=9)

scaler = MinMaxScaler()
# 시계열을 선회하면서 피팅합니다
for ss in range(num_sequence):
    scaler.partial_fit(samsung_x_train[:, ss, :]) #fit은 train data만 함

# 스케일링(변환)합니다
# 1. train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_train = np.concatenate(results, axis=1)

# 2. test data
# 다시 선언 => sampe data 수가 (train과) 다르니까??
num_sample   = samsung_x_test.shape[0] # 샘플 데이터 수 
num_sequence = samsung_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_x_test.shape[2] # Feature 수 (=9)

results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_test = np.concatenate(results, axis=1)

# 3. predict data
# 다시 선언 => sampe data 수가 (train과) 다르니까??
num_sample   = samsung_pred.shape[0] # 샘플 데이터 수
num_sequence = samsung_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_pred.shape[2] # Feature 수 (=9)

results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_pred = np.concatenate(results, axis=1)


# scaling for 비트
from sklearn.preprocessing import MinMaxScaler

num_sample   = bit_x_train.shape[0] # 샘플 데이터 수
num_sequence = bit_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_x_train.shape[2] # Feature 수 (=7)

scaler = MinMaxScaler()
# 시계열을 선회하면서 피팅합니다
for ss in range(num_sequence):
    scaler.partial_fit(bit_x_train[:, ss, :]) #fit은 train data만 함

# 스케일링(변환)합니다
# 1. train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_x_train = np.concatenate(results, axis=1)

# 2. test data
# 다시 선언 => sampe data 수가 (train과) 다르니까??
num_sample   = bit_x_test.shape[0] # 샘플 데이터 수 
num_sequence = bit_x_test.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_x_test.shape[2] # Feature 수 (=7)

results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_x_test = np.concatenate(results, axis=1)

# 3. validation data
# 다시 선언 => sampe data 수가 (train과) 다르니까??
num_sample   = bit_pred.shape[0] # 샘플 데이터 수
num_sequence = bit_pred.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_pred.shape[2] # Feature 수 (=7)

results = []
for ss in range(num_sequence):
    results.append(scaler.transform(bit_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
bit_pred = np.concatenate(results, axis=1)


# print(bit_x.shape) #(621, 5, 7)
# print(bit_y.shape) #(621,)
# print(samsung_x.shape) #(621, 5, 9)
# print(samsung_y.shape) #(621,)
# print(bit_x_train.shape)
# print(samsung_x_train.shape)

####################### 2.모델링 => LSTM(2차원 인풋), RNN(2차원 인풋), Conv1D(2차원 인풋) ################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D

#모델 1.
input1 = Input(shape=(5,9)) #입력 1
layer1_1 = LSTM(64, activation='relu', return_sequences=True)(input1)
layer1_2 = LSTM(128, activation='relu')(layer1_1)
layer1_3 = Dense(128, activation='relu')(layer1_2)
layer1_4 = Dropout(0.3)(layer1_3)
output1 = Dense(256, activation='relu')(layer1_4)


#모델 2.
input2 = Input(shape=(5,7)) #입력 2
layer2_1 = Conv1D(64, (5), padding="same")(input2)
layer2_2 = MaxPooling1D(pool_size=2)(layer2_1)
layer2_3 = Dropout(0.3)(layer2_2)
layer2_4 = Conv1D(128, (3), padding="same")(layer2_3)
layer2_5 = MaxPooling1D(pool_size=2)(layer2_4)
layer2_6 = Dropout(0.3)(layer2_5)
layer2_7 = Flatten()(layer2_6)
output2 = Dense(256, activation='relu')(layer2_7)


############### 모델 병합, concatenate ###################
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1,output2])
middle1 = Dense(32, activation='relu')(merge1)
middle1 = Dense(64, activation='relu')(middle1)
middle1 = Dense(128, activation='relu')(middle1)

################ output 모델 구성 #####################
output = Dense(64, activation='relu')(middle1)
output = Dense(1)(output) #출력 1

#모델 정의
model = Model(inputs=[input1,input2],
              outputs=output)
# model.summary()


####################### #3. 컴파일, 훈련 ################################
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
modelpath = './model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit([samsung_x_train,bit_x_train],samsung_y_train,epochs=1000,batch_size=16,verbose=2,
        callbacks=[es,cp], validation_split=0.2) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/samsung.hdf5')


#4. 평가
loss,mae = model.evaluate([samsung_x_test,bit_x_test],
                        samsung_y_test,
                        batch_size=16)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
result = model.predict([samsung_pred,bit_pred])
print("삼성 주가 예측값 : ", result)
