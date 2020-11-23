#Day11
#2020-11-23

#winequality-white.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터
#pandas로 csv 불러오기
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
count_data.plot()
plt.show() #분포가 특정 데이터에 치우쳐 있음

#그냥 훈련시키면 데이터 많은 쪽으로 훈련이 될 수도 있음 (상대적으로 quality 3,4,7,8,9가 적음)
#항상 데이터가 많은 쪽으로 (치우쳐 있으면) 머신이 판단할 확률이 큼
#그래서 acc가 높게 안나옴
#인위적으로 y의 labeling 값을 조절 (분포를 더 작게 잡아주기)


