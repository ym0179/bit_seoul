#Day9
#2020-11-19

# 나머지 데이터 셋을 저장하시오 (6개 irism,mnist 제외)
import numpy as np 

# cifar10
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy', arr=x_train)
np.save('./data/cifar10_x_test.npy', arr=x_test)
np.save('./data/cifar10_y_train.npy', arr=y_train)
np.save('./data/cifar10_y_test.npy', arr=y_test)

# fashion_mnist
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.save('./data/fashion_mnist_x_train.npy', arr=x_train)
np.save('./data/fashion_mnist_x_test.npy', arr=x_test)
np.save('./data/fashion_mnist_y_train.npy', arr=y_train)
np.save('./data/fashion_mnist_y_test.npy', arr=y_test)

# cifar100
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy', arr=x_train)
np.save('./data/cifar100_x_test.npy', arr=x_test)
np.save('./data/cifar100_y_train.npy', arr=y_train)
np.save('./data/cifar100_y_test.npy', arr=y_test)

# boston
from sklearn.datasets import load_boston
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
np.save('./data/boston_x.npy', arr=x_data)
np.save('./data/boston_y.npy', arr=y_data)

# diabetes
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
np.save('./data/diabetes_x.npy', arr=x_data)
np.save('./data/diabetes_y.npy', arr=y_data)

# breast_cancer
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
np.save('./data/cancer_x.npy', arr=x_data)
np.save('./data/cancer_y.npy', arr=y_data)

