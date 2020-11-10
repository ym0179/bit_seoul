#Multi-Layer Perceptron 다층 퍼셉트론

'''
- 인공 신경망에서 뉴런의 역할을 하는 기본 단위를 퍼셉트론이라고 부름
- 퍼셉트론은 다수의 입력(여러 신호)를 받아 하나의 값으로 출력
- 입력받은 신호를 합산할 때는 각 신호의 중요도에 따라 가중치를 부여해서 합산

퍼셉트론은 여러 신호가 주어졌을 때 어떤 입력 신호가 더 중요한지를 의미하는 가중치, 
입력 신호의 강도가 어느 정도 이상일 때 반응할 것인지를 의미하는 편향,
그리고 활성화 함수를 통해 어떤 출력을 내보낼 것인지를 결정하는 하나의 개체
'''

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array([range(101,201), range(311,411), range(100)])
x = np.transpose(x)
y = np.transpose(y)
# print(x)
print(x.shape) # (100,3)
print(y.shape)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size = 0.7, test_size = 0.2)
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train, y_train, train_size = 0.7)


# #2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()
# # model.add(Dense(3, input_dim = 1))
# model.add(Dense(3, input_shape = (1,))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(7))
# model.add(Dense(1))


# #3. 컴파일, 훈련
# model.compile(loss="mse", optimizer="adam", metrics=["mae"])
# # model.fit(x_train,y_train,validation_split=0.2,batch_size=1,epochs=100)
# # model.fit(x_train,y_train,batch_size=1,epochs=100)
# model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=1,epochs=100)


# #4. 평가,예측
# loss = model.evaluate(x_test,y_test,batch_size=1)
# print("loss: ",loss)
# y_pred = model.predict(x_test)
# # print("결과: \n",y_pred)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print("R2 : ",r2) # max 값: 1
