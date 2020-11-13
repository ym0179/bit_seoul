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


# def split_x(seq, size):
#     aaa = [] #는 테스트
#     for i in range(len(seq)-size+1):
#         subset = seq[i:(i+size)]

#         #aaa.append 줄일 수 있음
#         #소스는 간결할수록 좋다
#         # aaa.append([item for item in subset])
#         aaa.append(subset)
        
#     print(type(aaa))
#     return np.array(aaa)


'''
10개의 데이터에 size 5개
10년치 주가데이터, y값은 어디? -> 모델을 만들려면 x, y 반복적으로 머신에게 교육시켜야 하는데
시계열 데이터는 y값이 없음
1 2 3 4 5 6 7 8 9 10 (10일치 온도 데이터라고 가정)
그럼 11일째는 11도? 아님 9도일 수도 있음 명확하지 않음
모델을 짜려면 저 데이터를 x값과 y값으로 나눠 줘야
5일치를 잡아서 6일째 되는 날을 y값으로 잡겠다


데이터를 하나라도 더 아껴 써야
1 2 3 4 5 
2 3 4 5 6
3 4 5 6 7
...
6 7 8 9 10
7 8 9 10 ? < 여기서부턴 프레딕트해야
  x   | y 
5행 4열 ->  5, 4, 1 reshape y는 하나니까 1, 
2개씩 하는 것도 괜찮음 5 4 1 이면 5 2 2 (전체 data개수는 맞아야 한다)
주가, 환율, 금리(column 3개) -> 그럼 더 쪼개서 넣어야 함...
'''