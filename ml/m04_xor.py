#Day11
#2020-11-23

#인공지능의 겨울
#XOR 문제 해결하기
#비선형 모델로 해결

from sklearn.svm import LinearSVC #선형 서포트벡터 머신
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 데이터
# XOR 문제 => NonLinearly Separable Data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC()
model = SVC() # default: C=1, kernel='rbf', gamme='auto'

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
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과  [0 1 1 0]
model.score :  1.0
acc :  1.0

score = acc (분류문제에서만)
'''