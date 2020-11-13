#Day5
#2020-11-13
#숙제
#scaler 정리

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
           [100,200,300] #(14,3)
        ])
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])
x_pred = array([55,65,75])  #(3,)
x_pred = x_pred.reshape(1,3)   #(1,3)
x_pred2 = array([6600,6700,6800])  #(3,)

'''
데이터 전처리 - scaling

MinMaxScaler: 
- 모든 feature가 0과 1사이에 위치하게 만듬
- 최대/최소값이 각각 1, 0
- 이상치에 매우 민감
- Xscale = (X-Xmin) / (Xmax-Xmin)

StandardScaler: 
- 기본 스케일
- 범위를 정규 분포로 변환
- 모든 특성들이 같은 스케일 가짐
- 평균과 표준편차 사용
- 각 feature의 평균을 0, 분산을 1로 변경
- 이상치에 민감
- Xscale = (X-X_mean) / (X_standard deviation)

RobustScaler: (잘 안씀)
- 모든 특성들이 같은 크기를 갖는다는 점에서 StandardScaler와 비슷
- StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포
- 평균과 분산 대신 중앙값(median)과 IQR(interquartile range) 사용
- 이상치의 영향을 최소화

MaxAbsScaler: (잘 안씀)
- 최대절대값과 0이 각각 1, 0이 되도록 스케일링
- 양수 데이터로만 구성
- 큰 이상치에 민감

*모든 스케일링은 테스트 데이터가 포함된 전체 데이터셋이 아닌 오로지 훈련 데이터에 대해서만 fit되어야함
*이후 훈련 데이터와 테스트 데이터 각각을 스케일링 (scaler.transform())
*일반적으로 타겟(y) 데이터에 대한 스케일링은 진행하지 않음
*많은 스케일러들이 이상치의 영향을 많이 받는다 => 그나마 영향을 최소화한 RobustScaler가 있지만, 먼저 이상치를 제거해주는 것이 훨씬 좋음
'''
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler3 = RobustScaler()
scaler4 = MaxAbsScaler()

scaler1.fit(x) #fit은 train data만 함
x = scaler1.transform(x)
x_pred = scaler1.transform(x_pred)
# x = scaler.fit_transform(x)

#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(40, activation = 'relu', input_shape = (3,)))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='loss', patience=10)
#Tensorboard
to_hist = TensorBoard(log_dir="graph",
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

history = model.fit(
    x_train, y_train,
    callbacks=[es,to_hist],
    validation_data=(x_val,y_val),
    epochs=500, batch_size=1,
    verbose=2
)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_pred = model.predict(x_pred)
print("예측값: ", y_pred)