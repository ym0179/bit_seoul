#Day10
#2020-11-20

import pandas as pd
import numpy as np

####################### 데이터 불러오기 ################################
bit_x=np.load('./data/bit_x.npy', allow_pickle=True)
bit_y=np.load('./data/bit_y.npy', allow_pickle=True)
bit_pred=np.load('./data/bit_pred.npy', allow_pickle=True)
samsung_x=np.load('./data/samsung_x.npy', allow_pickle=True)
samsung_y=np.load('./data/samsung_y.npy', allow_pickle=True)
samsung_pred=np.load('./data/samsung_pred.npy', allow_pickle=True)
gold_x=np.load('./data/gold_x.npy', allow_pickle=True)
gold_y=np.load('./data/gold_y.npy', allow_pickle=True)
gold_pred=np.load('./data/gold_pred.npy', allow_pickle=True)
kosdaq_x=np.load('./data/kosdaq_x.npy', allow_pickle=True)
kosdaq_y=np.load('./data/kosdaq_y.npy', allow_pickle=True)
kosdaq_pred=np.load('./data/kosdaq_pred.npy', allow_pickle=True)


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
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8, random_state=77)
bit_x_train, bit_x_test, bit_y_train, bit_y_test = train_test_split(bit_x, bit_y, train_size=0.8, random_state=77)
gold_x_train, gold_x_test, gold_y_train, gold_y_test = train_test_split(gold_x, gold_y, train_size=0.8, random_state=77)
kosdaq_x_train, kosdaq_x_test, kosdaq_y_train, kosdaq_y_test = train_test_split(kosdaq_x, kosdaq_y, train_size=0.8, random_state=77)


samsung_pred = samsung_pred.reshape(1,samsung_pred.shape[0],samsung_pred.shape[1]) #pred 값 3차원으로 맞추기
bit_pred = bit_pred.reshape(1,bit_pred.shape[0],bit_pred.shape[1]) #pred 값 3차원으로 맞추기
gold_pred = gold_pred.reshape(1,gold_pred.shape[0],gold_pred.shape[1]) #pred 값 3차원으로 맞추기
kosdaq_pred = kosdaq_pred.reshape(1,kosdaq_pred.shape[0],kosdaq_pred.shape[1]) #pred 값 3차원으로 맞추기


def scale_fit(train_data, scaler):
    num_sample   = train_data.shape[0] # 샘플 데이터 수
    num_sequence = train_data.shape[1] # 시계열 데이터 수
    num_feature  = train_data.shape[2] # Feature 수

    # 시계열을 선회하면서 피팅
    for ss in range(num_sequence):
        scaler.partial_fit(train_data[:, ss, :]) #fit은 train data만 함

    # 스케일링(변환)
    # 1. train data
    num_sample   = train_data.shape[0] # 샘플 데이터 수
    num_sequence = train_data.shape[1] # 시계열 데이터 수
    num_feature  = train_data.shape[2] # Feature 수
    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(train_data[:, ss, :]).reshape(num_sample, 1, num_feature))
    train_data = np.concatenate(results, axis=1)

    return train_data, scaler

def scale_transform(data, scaler):
    num_sample   = data.shape[0] # 샘플 데이터 수
    num_sequence = data.shape[1] # 시계열 데이터 수
    num_feature  = data.shape[2] # Feature 수
    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(data[:, ss, :]).reshape(num_sample, 1, num_feature))
    data = np.concatenate(results, axis=1)

    return data


# 3차원 scaling
from sklearn.preprocessing import StandardScaler
# scaling for 삼성
samsung_scaler = StandardScaler()
samsung_x_train, samsung_scaler = scale_fit(samsung_x_train,samsung_scaler)
samsung_x_test = scale_transform(samsung_x_test,samsung_scaler)
samsung_pred = scale_transform(samsung_pred,samsung_scaler)
# scaling for 비트
bit_scaler = StandardScaler()
bit_x_train, bit_scaler = scale_fit(bit_x_train,bit_scaler)
bit_x_test = scale_transform(bit_x_test,bit_scaler)
bit_pred = scale_transform(bit_pred,bit_scaler)
# scaling for 금현물
gold_scaler = StandardScaler()
gold_x_train, gold_scaler = scale_fit(gold_x_train,gold_scaler)
gold_x_test = scale_transform(gold_x_test,gold_scaler)
gold_pred = scale_transform(gold_pred,gold_scaler)
# scaling for 코스닥
kosdaq_scaler = StandardScaler()
kosdaq_x_train, kosdaq_scaler = scale_fit(kosdaq_x_train,kosdaq_scaler)
kosdaq_x_test = scale_transform(kosdaq_x_test,kosdaq_scaler)
kosdaq_pred = scale_transform(kosdaq_pred,kosdaq_scaler)


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
layer1_1 = LSTM(128, activation='relu')(input1)
# layer1_2 = LSTM(32, activation='relu')(layer1_1)
layer1_2 = Dense(128, activation='relu')(layer1_1)
layer1_3 = Dropout(0.3)(layer1_2)
output1 = Dense(32, activation='relu')(layer1_3)


#모델 2.
input2 = Input(shape=(5,6)) #입력 2
layer2_1 = LSTM(32, activation='relu', return_sequences=True)(input1)
layer2_2 = LSTM(16, activation='relu')(layer2_1)
layer2_3 = Dense(256, activation='relu')(layer2_2)
layer2_4 = Dropout(0.3)(layer2_3)
# layer2_5 = Dense(128, activation='relu')(layer2_4)
output2 = Dense(64, activation='relu')(layer2_4)

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
layer3_1 = LSTM(64, activation='relu')(input3)
layer3_2 = Dense(256, activation='relu')(layer3_1)
layer3_3 = Dropout(0.3)(layer3_2)
output3 = Dense(32, activation='relu')(layer3_3)

#모델 4.
input4 = Input(shape=(5,3)) #입력 4
layer4_1 = LSTM(128, activation='relu')(input4)
layer4_2 = Dense(64, activation='relu')(layer4_1)
layer4_3 = Dropout(0.3)(layer4_2)
# layer4_4 = Dense(128, activation='relu')(layer4_3)
output4 = Dense(32, activation='relu')(layer4_3)

############### 모델 병합, concatenate ###################
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1,output2,output3,output4])
middle1 = Dense(512, activation='relu')(merge1)
middle1 = Dense(128, activation='relu')(middle1)

################ output 모델 구성 #####################
output = Dense(64, activation='relu')(middle1)
output = Dense(1)(output) #출력 1

#모델 정의
model = Model(inputs=[input1,input2,input3,input4],
              outputs=output)
# model.summary()


####################### #3. 컴파일, 훈련 ################################
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
modelpath = './model/samsung2.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
model.fit([samsung_x_train,bit_x_train,gold_x_train,kosdaq_x_train],samsung_y_train,epochs=1000,batch_size=32,verbose=2,
        callbacks=[es,cp], validation_split=0.2) 

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./model/samsung2.hdf5')


#4. 평가
loss,mae = model.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],
                        samsung_y_test,
                        batch_size=32)
print("loss : ",loss)
print("mae : ",mae)

#5. 예측
result = model.predict([samsung_pred,bit_pred,gold_pred,kosdaq_pred])
print("삼성 시가 예측값 : ", result)

'''
loss :  868765.5625
mae :  734.5729370117188
삼성 시가 11/23 예측값 :  [65034.03]
삼성 시가 11/20 예측값 :  [64390.043]
삼성 시가 11/20 실제값 : 63,900
R2 :  0.9774594110034163
'''