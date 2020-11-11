#Day3
#2020-11-11
'''
<앙상블>
예측 모형을 통합해서 하나의 예측을 수행
서로 다른 예측 모형들을 합쳐 더 강한 예측 모형을 만들 수 있음
앙상블 기법에는 배깅(Bagging), 부스팅(Boosting), 보팅(Voting), 스태킹(Stacking) 등 여러가지가 존재
단일 모델의 약점(높은 overfitting 가능성)을 보완하고, 
궁극적으로 일반화 성능(generalization capacity)를 향상시키는 목적
'''
#1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(711,811), range(100)])
x2 = np.array([range(4,104), range(761,861), range(100)])

y1 = np.array([range(101,201), range(311,411), range(100)])
y2 = np.array([range(501,601), range(431,531), range(100,200)])
y3 = np.array([range(501,601), range(431,531), range(100,200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

#train test split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1,train_size=0.7,shuffle=True)
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x2,y2,y3,train_size=0.7,shuffle=True)
# y3_train, y3_test = train_test_split(y3,train_size=0.7,shuffle=True)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#모델 1.
input1 = Input(shape=(3,)) #입력 1
dense1_1 = Dense(5, activation='relu',name='dense1_1')(input1)
dense1_2 = Dense(5, activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(4, activation='relu',name='dense1_3')(dense1_2)
dense1_4 = Dense(3, activation='relu',name='dense1_4')(dense1_3)
output1 = Dense(3, activation='linear',name='dense1_5')(dense1_4)
model1 = Model(inputs=input1,outputs=output1)
# model1.summary()

#모델 2.
input2 = Input(shape=(3,)) #입력 2
dense2_1 = Dense(15, activation='relu',name='dense2_1')(input2)
dense2_2 = Dense(7, activation='relu',name='dense2_2')(dense2_1)
dense2_3 = Dense(3, activation='relu',name='dense2_3')(dense2_2)
output2 = Dense(3, activation='linear',name='dense2_4')(dense2_3)
model2 = Model(inputs=input2,outputs=output2)
# model2.summary()

############### 모델 병합, concatenate
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#연산하지 않고 병합만 함 (병합 후 앞에 레이어 받아서 다음 레이어와 연산)
# merge1 = concatenate([output1,output2]) #concatenate로 모델 merge
merge1 = Concatenate(axis=-1)([output1,output2]) 
#axis=0이면 행 기준(같은 column,row더해줌), axis=1이면 열 기준(column늘어남)
#axis=-1 default => axis=1 과 같음
middle1 = Dense(30, activation='relu',name='middle1')(merge1) #변수명 같아도 무방
middle1 = Dense(7, activation='relu', name='middle2')(middle1)
middle1 = Dense(11, name='middle3')(middle1)

################ output 모델 구성 (분기)
output1 = Dense(30, activation='relu',name='output1_1')(middle1)
output1 = Dense(7, activation='relu',name='output1_2')(output1)
output1 = Dense(7, activation='relu',name='output1_3')(output1)
output1 = Dense(3,name='output1_4')(output1) #출력 1

output2 = Dense(15, activation='relu',name='output2_1')(middle1)
output2_1 = Dense(14, activation='relu',name='output2_2')(output2)
output2_2 = Dense(11, activation='relu',name='output2_3')(output2_1)
output2_3 = Dense(11, activation='relu',name='output2_4')(output2_2)
output2_4 = Dense(3,name='output2_5')(output2_3) #출력 2

output3 = Dense(10, activation='relu',name='output3_1')(middle1)
output3_1 = Dense(10, activation='relu',name='output3_2')(output3)
output3_2 = Dense(15, activation='relu',name='output3_3')(output3_1)
output3_3 = Dense(3,name='output3_4')(output3_2) #출력 2

#총 5개의 모델을 합침
#모델 정의
model = Model(inputs=[input1,input2],
              outputs=[output1,output2_4,output3_3])
model.summary()


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit([x1_train,x2_train],
          [y1_train,y2_train,y3_train],
          validation_split=0.25,
          epochs=100,batch_size=8,
          verbose=2)


#4. 평가,예측
result = model.evaluate([x1_test,x2_test],
                        [y1_test,y2_test,y3_test],
                        batch_size=8)
print("result : ",result)
"""
모델의 output이 2개일 때 5개의 값이 나옴
1.전체 loss
2.첫번째output의 loss
3.두번째output의 loss
4.첫번째output의 mse
5.두번째output의 mse
"""

