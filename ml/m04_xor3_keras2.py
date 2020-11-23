#Day11
#2020-11-23

#인공지능의 겨울
#XOR 문제 해결하기
#keras로 해결 -> 이진분류, layer 여러개 -> acc: 1.0

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
model.add(Dense(30, input_dim=2, activation='relu')) #이진 분류
model.add(Dense(20, activation='relu')) #이진 분류
model.add(Dense(1, activation='sigmoid')) #이진 분류

# 3. 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(x_data,y_data,batch_size=1,epochs=100)

# 4. 평가, 예측
acc1 = model.evaluate(x_data,y_data)
print("model.evaluate : ", acc1)

y_predict = model.predict(x_data) #y 데이터가 나오는지 확인
print(x_data, "의 예측 결과 ", y_predict.T)

y_predict = np.round(y_predict, 0) #y_predict가 실수라서, 그걸 반올림/내림 해서 정수로 바꾸기
acc2 = accuracy_score(y_data,y_predict)
print("acc : ", acc2)

'''
model.evaluate :  [0.22735607624053955, 1.0]
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과  [[0.3360899  0.90118873 0.82060117 0.17967701]]
acc :  1.0
'''