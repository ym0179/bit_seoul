#Day11
#2020-11-23

#인공지능의 겨울
#XOR 문제 해결하기
#keras로 해결? -> 이진분류, layer 하나 -> acc: 0.25 / 0.5

from sklearn.svm import LinearSVC 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# XOR 문제 => NonLinearly Separable Data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC()
# model = SVC() # default: C=1, kernel='rbf', gamme='auto'
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) #이진 분류

# 3. 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(x_data,y_data,batch_size=1,epochs=100)

# 4. 평가, 예측
acc1 = model.evaluate(x_data,y_data)
print("model.evaluate : ", acc1)

y_predict = model.predict(x_data) #y 데이터가 나오는지 확인
print(x_data, "의 예측 결과 ", y_predict.T)

y_predict = np.round(y_predict, 0)
acc2 = accuracy_score(y_data,y_predict)
print("acc : ", acc2)

'''
model.evaluate :  [0.7900441288948059, 0.25]
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과  [[0.52993727 0.24439357 0.45918107 0.19587578]]
acc :  0.25

model.evaluate :  [0.7113118171691895, 0.5]
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과  [[0.49365777 0.6484734  0.40457463 0.56248814]]
acc :  0.5
'''