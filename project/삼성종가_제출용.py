#Day10
#2020-11-20

import pandas as pd
import numpy as np

####################### 데이터 불러오기 ################################

#저장한 데이터 불러오기
bit_x=np.load('./data/bit_x.npy',allow_pickle=True)
bit_y=np.load('./data/bit_y.npy',allow_pickle=True)
bit_pred=np.load('./data/bit_pred.npy',allow_pickle=True)

samsung_x=np.load('./data/samsung_x.npy',allow_pickle=True)
samsung_y=np.load('./data/samsung_y.npy',allow_pickle=True)
samsung_pred=np.load('./data/samsung_pred.npy',allow_pickle=True)

####################### 1.데이터 전처리 ################################
#float type 변환
bit_x = bit_x.astype('float32')
bit_y = bit_y.astype('float32')
bit_pred = bit_pred.astype('float32')
samsung_x = samsung_x.astype('float32')
samsung_y = samsung_y.astype('float32')
samsung_pred = samsung_pred.astype('float32')

#pred 값 3차원으로 맞추기
samsung_pred = samsung_pred.reshape(1,samsung_pred.shape[0],samsung_pred.shape[1])
bit_pred = bit_pred.reshape(1,bit_pred.shape[0],bit_pred.shape[1])

#train-test split
from sklearn.model_selection import train_test_split
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test, bit_y_train, bit_y_test = train_test_split(bit_x, bit_y, train_size=0.8)


# scaling for 삼성 - 3차원 시계열 축을 순회하면서 스케일링 (시계열이 포함된 3차원 데이터 스케일링)
from sklearn.preprocessing import MinMaxScaler
num_sample   = samsung_x_train.shape[0] # 샘플 데이터 수
num_sequence = samsung_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = samsung_x_train.shape[2] # Feature 수 (=9)

scaler = MinMaxScaler()
# 시계열을 선회하면서 피팅
for ss in range(num_sequence):
    scaler.partial_fit(samsung_x_train[:, ss, :]) #fit은 train data만 함

# 스케일링(변환)
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
num_sample   = bit_x_train.shape[0] # 샘플 데이터 수
num_sequence = bit_x_train.shape[1] # 시계열 데이터 수 (=5)
num_feature  = bit_x_train.shape[2] # Feature 수 (=7)

scaler = MinMaxScaler()
# 시계열을 선회하면서 피팅
for ss in range(num_sequence):
    scaler.partial_fit(bit_x_train[:, ss, :]) #fit은 train data만 함

# 스케일링(변환)
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

####################### 2.모델링 => LSTM(2차원 인풋), RNN(2차원 인풋), Conv1D(2차원 인풋) ################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D

#모델 1.
input1 = Input(shape=(5,9)) #입력 1
layer1_1 = LSTM(128, activation='relu')(input1)
layer1_2 = Dense(128, activation='relu')(layer1_1)
layer1_3 = Dropout(0.3)(layer1_2)
output1 = Dense(256, activation='relu')(layer1_3)


#모델 2.
input2 = Input(shape=(5,7)) #입력 2
layer2_1 = Conv1D(256, (2), padding="same")(input2)
layer2_2 = MaxPooling1D(pool_size=2)(layer2_1)
layer2_3 = Dropout(0.3)(layer2_2)
layer2_4 = Conv1D(128, (2), padding="same")(layer2_3)
layer2_5 = MaxPooling1D(pool_size=2)(layer2_4)
layer2_6 = Dropout(0.3)(layer2_5)
layer2_7 = Flatten()(layer2_6)
output2 = Dense(256, activation='relu')(layer2_7)


############### 모델 병합, concatenate ###################
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1,output2])
middle1 = Dense(128, activation='relu')(merge1)
middle1 = Dropout(0.3)(middle1)
middle1 = Dense(64, activation='relu')(middle1)

################ output 모델 구성 #####################
output = Dense(32, activation='relu')(middle1)
output = Dense(1)(output) #출력 1

#모델 정의
model = Model(inputs=[input1,input2],
              outputs=output)
# model.summary()


####################### #3. 컴파일, 훈련 ################################

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=50,mode='auto')
modelpath = './model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit([samsung_x_train,bit_x_train],samsung_y_train,epochs=1000,batch_size=1,verbose=2,
        callbacks=[es,cp], validation_split=0.2) 

# 모델 불러오기
from tensorflow.keras.models import load_model
# model = load_model('./model/samsung-62-2391732.7500.hdf5')


#4. 평가
loss,mae = model.evaluate([samsung_x_test,bit_x_test],
                        samsung_y_test,
                        batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
result = model.predict([samsung_pred,bit_pred])
print("삼성 주가 예측값 : ", result)
