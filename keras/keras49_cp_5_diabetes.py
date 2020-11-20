#Day8
#2020-11-18

#당뇨병 진행도 예측 모델 학습 (sklearn 라이브러리)
'''
[x]
442 행 10 열 
Age
Sex
Body mass index
Average blood pressure
S1
S2
S3
S4
S5
S6

[y]
442 행 1 열 
target: a quantitative measure of disease progression one year after baseline
'''

from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(442, 10) (442,)
# print(dataset)

# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x) # fit하고
# x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자

#1. 전처리
#train-test split -> scaling train 만 하니까 젤 먼저 split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state = 44)

#scaling - 2차원 input만 가능 -> reshape 나중에
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit은 train data만 함
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_pred = x_test[:10]
y_pred = y_test[:10]

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# model = Sequential()
# model.add(Dense(64, activation='relu',input_shape=(10,)))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))

model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(10,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/diabetes-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(monitor='val_loss',patience=70,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es, cp],validation_split=0.3) 

model.save('./save/model_diabetes.h5')
model.save_weights('./save/weights_diabetes.h5')

#4. 평가
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("mae : ",mae)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) #인치 단위
#1번째 그림
plt.subplot(2, 1, 1) #2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #plt.plot에서 명시한 label이 박스형태로 상단 오른쪽에 나옴

#2번째 그림
plt.subplot(2, 1, 2) #2행 1열 중 두번째
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()

plt.title('MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['mae','val_mae']) #location명시 안하면 알아서 빈자리에 박스 그림

plt.show()

#5. 예측
result = model.predict(x_pred)
print("예측값 : ", result.T.reshape(10,)) #보기 쉽게
print("실제값 : ", y_pred)

y_predicted =  model.predict(x_test) #x_pred 10개밖에 없음응로 x_test 가지고 RMSE, R2 계산

#RMSE
#R2
import numpy as np
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predicted):
    return np.sqrt(mean_squared_error(y_test,y_predicted))
print("RMSE : ", RMSE(y_test, y_predicted))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predicted)
print("R2 : ",r2) # max 값: 1

'''
loss :  1894.115234375
mae :  33.49161148071289
예측값 :  [233.46266   79.84389   72.59121  190.11183   85.084595 210.47282
  84.32935  132.74821  146.85315   84.45865 ]
실제값 :  [277.  77.  94. 248.  55. 235.  89. 127.  90.  63.]
RMSE :  43.52143906372869
R2 :  0.6893922268591381
'''
