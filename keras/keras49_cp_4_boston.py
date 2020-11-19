#Day8
#2020-11-18

#보스턴 집값 예측: 1978년에 발표된 데이터로 미국 보스턴 지역의 주택 가격에 영향을 미치는 요소들을 정리


from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(506, 13) (506,)

#1. 전처리
#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state = 44)

#scaling
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
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# model.add(Dense(32, activation='relu',input_shape=(13,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/boston-{epoch:02d}-{val_loss:.4f}.hdf5' #모델 폴더 만들어 주기
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto') #val_loss 가장 좋은거 저장
hist = model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es,cp],validation_data=(x_val,y_val)) 

model.save('./save/model_boston.h5')
model.save_weights('./save/weights_boston.h5')

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
loss :  4.049808979034424
mae :  1.4133822917938232
예측값 :  [19.611595 16.140156 18.706654 23.607738 13.600585 29.706987 24.058996
 16.93112  24.74956  14.395992]
실제값 :  [20.  16.1 18.7 22.9 13.4 30.8 23.7 16.2 24.1 11.9]
RMSE :  2.0124135091760937
R2 :  0.955160030688846
'''