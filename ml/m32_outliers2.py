#Day21
#2020-12-07

#이상치 제거
#outliers1을 행렬형태에도 적용할 수 있도록 수정
# percentile

import numpy as np

def outliers(data_out, column) :
    column_data = data_out[:,column]
    print(column_data)

    quartile_1, quartile_3 = np.percentile(column_data, [25,75]) #데이터의 25%, 75%지점
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outlier_index = np.where((column_data > upper_bound) | (column_data < lower_bound))
    print("outlier_index : ", outlier_index)
    outlier_value = column_data[(column_data > upper_bound) | (column_data < lower_bound)]
    print("outlier_value : ", outlier_value)

    for i in range(len(outlier_value)) :
        data_out = data_out[data_out[:,column] != outlier_value[i]]
    return data_out

def outliers2(data_out) :
    n,m = data_out.shape

    for i in range(m): 
        column_data = data_out[:,i]
        print(column_data)

        quartile_1, quartile_3 = np.percentile(column_data, [25,75]) #데이터의 25%, 75%지점
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outlier_index = np.where((column_data > upper_bound) | (column_data < lower_bound))
        print("outlier_index : ", outlier_index)
        outlier_value = column_data[(column_data > upper_bound) | (column_data < lower_bound)]
        print("outlier_value : ", outlier_value)

        for j in range(len(outlier_value)) :
            data_out = data_out[data_out[:,i] != outlier_value[j]]
    print("data_out : ", data_out)
    return data_out

a = np.array([[1,2,3,4,10000,6,7,5000,90,100],
              [10000,20000,-100000,40000,50000,60000,70000,8,90000,100000]])
a = a.transpose()
# print("a : ", a)
# b = outliers(a,0)
outliers2(a)

# print("이상치의 위치 : ",b) #index 값


# https://lsjsj92.tistory.com/556


'''
def outliers(data_out) :
    data_out = data_out.T # 뒤집어서 빼기 편하게
    del_index = np.array([],dtype=np.int)
    for cnt in range(data_out.shape[1]):
        column_data = data_out[:,cnt]
        quartile_1, quartile_3 = np.percentile(column_data, [25,75]) #데이터의 25%, 75%지점
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outlier_index = np.where((column_data>upper_bound)|(column_data<lower_bound))
        del_index = np.append(del_index, outlier_index) # 인덱스를 모아서
    data_out = np.delete(data_out, del_index, axis=0) # 모은 인덱스를 빼라
    data_out = data_out.T # 다시 뒤집자
    return data_out
'''