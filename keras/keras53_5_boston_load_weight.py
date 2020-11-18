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


#2. 모델
################### 1. load_model ########################
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_boston.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result1[0])
print("mae : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/boston-119-1.0627.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result2[0])
print("mae : ", result2[1])


################ 3. load_weights ##################
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model3 = Sequential()
model3.add(Dense(64, activation='relu',input_shape=(13,)))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(10, activation='relu'))
model3.add(Dense(1))

#3. 컴파일, 훈련
model3.compile(loss="mse", optimizer="adam", metrics=["mae"])
model3.load_weights('./save/weights_boston.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test,y_test,batch_size=1)
print("loss : ", result3[0])
print("mae : ", result3[1])

'''
save 한 모델:
loss :  4.049808979034424
mae :  1.4133822917938232
'''
# model1
# loss :  4.049808979034424
# mae :  1.4133822917938232

# model2
# loss :  3.710859537124634
# mae :  1.2830994129180908

# model3
# loss :  4.049808979034424
# mae :  1.4133822917938232
