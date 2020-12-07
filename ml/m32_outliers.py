#Day21
#2020-12-07

#이상치 제거

import numpy as np

def outliers(data_out) :
    # 데이터 4등분
    quartile_1, quartile_3 = np.percentile(data_out, [25,75]) #데이터의 25%, 75%지점
    #전체 데이터의 길이의 1/4 지점, 3/4 지점
    print("1사분위 : ", quartile_1) #1사분위 :  3.25
    print("3사분위 : ", quartile_3) #3사분위 :  97.5

    #1,5배수
    iqr = quartile_3 - quartile_1 #94.25
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))


a = np.array([1,2,3,4,10000,6,7,5000,90,100])
b = outliers(a)
print("이상치의 위치 : ",b) #index 값