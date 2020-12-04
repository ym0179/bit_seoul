#Day20
#2020-12-04

#실습
#Embedding 빼고 lstm, conv1d 구성
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

x = token.texts_to_sequences(docs) #문장을 수치로 바꾼거 
# print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') #뒤 post
# print(pad_x) #[ 0  0  0  2  3] 이런식으로 0 채워줌
# print(pad_x.shape) #(12, 5)


word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size) #25

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D

pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)

model = Sequential()
# model.add(LSTM(32, input_shape=(pad_x.shape[1], 1)))
model.add(Conv1D(32, 3, padding='same',input_shape=(pad_x.shape[1], 1))) # 2=커널사이즈
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 3, padding='same')) 
model.add(Conv1D(32, 3, padding='same')) 
model.add(Dense(1, activation = 'sigmoid')) #scala

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)

