#Day20
#2020-12-04

#Flatten 확인

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', "참 최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다','한 번 더 보고 싶네요', '글쎄요',
        '별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밋네요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)

# print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, 
# '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, 
# '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24}
# print(len(token.word_index)) #24

x = token.texts_to_sequences(docs) #문장을 수치로 바꾼거 
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
# 0 을 채워서 같은 shape로 만들어줌 (가장 긴거 기준)
# 글은 시계열 데이터 -> 0을 채울때 앞에를 0을 채워주고 의미있는 숫자를 뒤로 보내줘야함

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') #뒤 post
print(pad_x) #[ 0  0  0  2  3] 이런식으로 0 채워줌
print(pad_x.shape) #(12, 5)


word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size) #25

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
# model.add(Embedding(25,10)) #input_length 안써도 parameter 숫자는 같다 / 명시하면 똑같이 넣어줘야하고 아니면 냅두고
model.add(Embedding(25,10,input_length=5)) #두가지 방식의 기법 #원핫인코딩 벡터화
# model.add(LSTM(32))
model.add(Flatten()) #input_length 명시 안하면 에러
model.add(Dense(1, activation = 'sigmoid')) #scala

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1] #metrics 반환
print("acc : ", acc)
