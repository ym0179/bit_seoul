#Day20
#2020-12-04

from tensorflow.keras.preprocessing.text import Tokenizer

# text = '나는 울트라 맛있는 밥을 먹었다'
text = '나는 진짜 맛있는 밥을 진짜 먹었다'


token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'나는': 1, '울트라': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5} => 진짜라는 단어가 많이 나와서 1번
print(len(token.word_index)) #5

x = token.texts_to_sequences([text]) #문장을 수치로 바꾼거 
print(x)
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# [[2, 1, 3, 4, 1, 5]]
# 원핫인코딩 / 라벨인코딩 (2가 1의 2배인가?)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size + 1) #원핫인코딩 0 빼고 1,2,3,4,5 임으로 총 6개
print(x)
