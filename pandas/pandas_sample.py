#Day9
#2020-11-19

import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)

data = randn(5,4) #5행 4열
print(data)
df = pd.DataFrame(data, index='A B C D E'.split(),
                columns='가 나 다 라'.split())
print(df)

data2 = [[1,2,3,4,], [5,6,7,8], [9,10,11,12], 
        [13,14,15,16], [17,18,19,20]] #list
df2 = pd.DataFrame(data2, index=['A','B','C','D','E'],
            columns=['가','나','다','라'])
print(df2)
#     가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20


df3 = pd.DataFrame(np.array([[1,2,3],[4,5,6]]))
print(df3)

print("df2['나'] :\n",df2['나']) #2,6,10,14,18
print("df2['나','라'] :\n",df2[['나','라']]) #2,6,10,14,18
                                            #4,8,12,16,20

# print("df2[0] : ", df2[0]) #에러, 컬럼명으로 해줘야 에러 안남
# print("df2.loc['나'] : \n", df2.loc['나']) #에러, loc 행에서만 사용 가능 (행과 함께 사용)

print("df2.iloc[:,2] : \n", df2.iloc[:, 2]) #3,7,11,15,19
# print("df2[:,2] : \n", df2[:, 2]) #에러


#행
print("df2.loc['A'] : \n", df2.loc['A']) #A행 출력
print("df2.loc['A','C'] : \n", df2.loc[['A','C']]) #A, C행 출력

print("df2.iloc[0] : \n", df2.iloc[0]) #A행 출력
print("df2.iloc[0,1] : \n", df2.iloc[[0,2]]) #A, C행 출력


#행렬
print("df2.loc[['A','B'], ['나','다']] : \n",df2.loc[['A','B'], ['나','다']])


#한개의 값만 확인
print("df2.loc['E','다'] : \n",df2.loc['E','다']) #19
print("df2.iloc[4,2] : \n",df2.iloc[4,2]) #19
print("df2.iloc[4][2] : \n",df2.iloc[4][2]) #19
