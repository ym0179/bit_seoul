#Day9
#2020-11-19

import pandas as pd
import numpy as np

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',')
# print(datasets)

# print(datasets.shape) #(150, 5)

# index_col = None, 0, 1 / header = None, 0, 1
# 실습, 위 경우의 수로 shape 결과치 도출
'''
header/index_col    None #index 추가         0 #0번째 행 index             1 #1번째 행 index
None #header 추가       (151, 6)                  (151, 5)                     (151, 5)
0 #0번째 열 index       (150, 6)                  (150, 5)                     (150, 5)
1 #1번째 열 index       (149, 6)                  (149, 5)                     (149, 5)   
'''

print(datasets.head()) #위에서 5개
print(datasets.tail()) #아래에서 5개
print(type(datasets)) #<class 'pandas.core.frame.DataFrame'>

#numpy는 전체가 수치화되있는 데이터만 가능, 전체 데이터 형태가 하나로 통일되어 있어야함
#pandas dataframe를 numpy 배열로 변환하기
# aaa = datasets.to_numpy()
aaa = datasets.values
print(aaa)
print(type(aaa)) #<class 'numpy.ndarray'>
print(aaa.shape) #(150, 5)

np.save('./data/iris_ys_pd.npy', arr=aaa)