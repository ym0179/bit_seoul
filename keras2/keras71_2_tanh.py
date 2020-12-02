#Day18
#2020-11-18

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5,5,0.1)
y =np.tanh(x)

# print("x : ", x)
# print("y : ", y)

plt.plot(x,y)
plt.grid()
plt.show()