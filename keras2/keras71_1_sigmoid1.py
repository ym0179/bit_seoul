#Day18
#2020-12-02

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) : #데이터가 0또는 1에 수렴
    return 1 / (1 + np.exp(-x))

x = np.arange(-5,5,0.1)
y = sigmoid(x)

# print("x : ", x)
# print("y : ", y)

plt.plot(x,y)
plt.grid()
plt.show()