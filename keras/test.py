import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
x2 = np.array([6,7,8,9,10])

model = Sequential() 
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=500, batch_size=1) 

loss, mse = model.evaluate(x, y, batch_size=1)

print("loss: ", loss)
print("mse: ", mse)

result = model.predict(x2)
print("predicted result: ", result)
#회귀 모델은 선형이기 때문에 딱 맞아떨어지지 않음 -> 소수점을 사용 = accuracy는 평가지표로 사용할 수 없음
#accuracy는 분류모델에서 사용