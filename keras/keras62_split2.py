#Day9
#2020-11-19

# <<과제>>
# 커스텀 스플릿함수 만든 것의 문제

# iris 데이터를
# 150,4를 5개씩 자른다면 146,5,4가 되어야 한다

# 예전꺼
# import numpy as np
# dataset = np.array(range(5,15))
# size = 3

# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size + 1):
#         subset = seq[i : (i+size)]
#         aaa.append([subset])
#     return np.array(aaa)

# dataset = split_x(dataset, size)
# print("===================")
# print(dataset)


import numpy as np
from sklearn.datasets import load_iris
test = load_iris()
x = test.data
print(x.shape) #(150, 4)

def split_x2(seq, size):
    aaa = []
    print("size :",size)
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x2(x, 5)
print("===================")
print(dataset)


# def split_x2(seq, size):
#     aaa = []
#     bbb = []
#     print("size :",size)
#     for i in range(len(seq) - size + 1):
#         aaa.clear()
#         for j in range(i, i+size):
#             subset = seq[j]
#             aaa.append(subset)
#         bbb.append(np.array(aaa))
#     return np.array(bbb)

# dataset = split_x2(x, 5)
# print("===================")
# print(dataset)