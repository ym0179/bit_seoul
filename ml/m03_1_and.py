#Day11
#2020-11-23

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
# AND 문제
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# 2. 모델
model = LinearSVC()

# 3. 훈련
model.fit(x_data,y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data) #y 데이터가 나오는지 확인
print(x_data, "의 예측 결과 ", y_predict)

acc1 = model.score(x_data,y_data)
print("model.score : ", acc1)

acc2 = accuracy_score(y_data,y_predict)
print("acc : ", acc2)

'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과  [0 0 0 1]
model.score :  1.0
acc :  1.0
'''