#Day4
#2020-11-12

import numpy as np
dataset = np.array(range(5,15))
size = 3

def split_x(seq, size):
    aaa = []
    print("size :",size)
    for i in range(len(seq) - size + 1):
        print("i :",i)
        subset = seq[i : (i+size)]
        print("subset :",subset)
        aaa.append([item for item in subset])
        print("type :",type(aaa))
        print("aaa :",aaa)
    return np.array(aaa)

datasets = split_x(dataset, size)
print("===================")
print(datasets)