#Day5
#2020-11-13

#모델 저장
import numpy as np

#모델을 구성하시오.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer

model = Sequential()
model.add(LSTM(30, activation='relu', input_length=4, input_dim=1)) # *****
model.add(Dense(50, activation='relu')) #default activation = linear
model.add(Dense(70, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dense(1)) #output: 1개
# output layer는 제외하고 저장


# model.summary()

#모델 저장: 경로 주의***** 
#모델의 확장자:h5
# model.save("save1.h5")
 
#\Study에 저장됨 (작업 폴더에) 즉, 비주얼스튜디오 코드 실행시 root폴더가 작업그룹이다 현재:study
#하지만 보기 편하도록 별도의 폴더를 만들겠다

#파일명에 n들어가면 \n <- 이렇게 돼서 개행 될 수도 있음 (그밖의 다른 이스케이프 문자들)
#주의!!
model.save("./save/keras26_model.h5") #. = root(최상위)
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")