#Day3
#2020-11-11

#How can I obtain reproducible results
#재현가능한 결과 얻기
import numpy as np
import tensorflow as tf
import random as python_random

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

#1. 데이터
# import numpy as np
x1 = np.array([range(1,101), range(711,811), range(100)])
x2 = np.array([range(4,104), range(761,861), range(100)])
y1 = np.array([range(101,201), range(311,411), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

x1_predict = np.array([(1,711,0)])
x2_predict = np.array([(4,761,0)])

# print(x_predict)

#train test split
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1,
    train_size=0.8)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#모델 1.
input1 = Input(shape=(3,)) #입력 1
dense1_1 = Dense(20, activation='relu',name='dense1_1')(input1)
dense1_2 = Dense(15, activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(10, activation='relu',name='dense1_3')(dense1_2)
output1 = Dense(7, activation='relu',name='dense1_4')(dense1_3)

#모델 2.
input2 = Input(shape=(3,)) #입력 2
dense2_1 = Dense(10, activation='relu',name='dense2_1')(input2)
dense2_2 = Dense(7, activation='relu',name='dense2_2')(dense2_1)
output2 = Dense(3, activation='relu',name='dense2_3')(dense2_2)

############### 모델 병합, concatenate
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1,output2])
middle1 = Dense(15, activation='relu',name='middle1')(merge1)
middle1 = Dense(10, activation='relu', name='middle2')(middle1)
middle1 = Dense(7, activation='relu', name='middle3')(middle1)

################ output 모델 구성
output1 = Dense(10, activation='relu',name='output1_1')(middle1)
output1 = Dense(3,name='output1_2')(output1) #출력 1

#모델 정의
model = Model(inputs=[input1,input2],
              outputs=output1)
# model.summary()


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit([x1_train,x2_train],
          y1_train,
          validation_split=0.25,
          epochs=500,batch_size=8,
          verbose=2)


#4. 평가,예측
result = model.evaluate([x1_test,x2_test],
                        [y1_test],
                        batch_size=8)
print("result : ",result)

predict = model.predict([x1_predict,x2_predict])
print("predict : ",predict) #101,311,0


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_pred = model.predict([x1_test,x2_test])

def RMSE(y1_test,y_pred):
    return np.sqrt(mean_squared_error(y1_test,y_pred))

r2 = r2_score(y1_test,y_pred)

print("RMSE : ",RMSE(y1_test,y_pred))
print("r2 : ",r2)

