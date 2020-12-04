#Day20
#2020-12-04

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('./data/dog_cat/강아지.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/고양이.jpg', target_size=(224,224))
img_lion = load_img('./data/dog_cat/라이언.jpg', target_size=(224,224))
img_suit = load_img('./data/dog_cat/수트.jpg', target_size=(224,224))
# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
# print(arr_dog)
# print(type(arr_dog)) #<class 'numpy.ndarray'>
# print(arr_dog.shape) #(341, 512, 3)

arr_cat = img_to_array(img_cat)
# print(arr_cat)
# print(type(arr_cat)) #<class 'numpy.ndarray'>
# print(arr_cat.shape) #(650, 435, 3)

arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

#VGG16 RGB -> BGR 형태로 바꿔주기
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
# print(arr_dog.shape) #(341, 512, 3)

arr_input = np.stack([arr_dog,arr_cat,arr_lion,arr_suit])
print(arr_input.shape) #(2, 224, 224, 3)

#2. 모델 구성
model = VGG16() #include_top = True (VGG 원래 input이 244,244), trainable = False
probs = model.predict(arr_input)
# print(probs)
# print('probs.shape : ',probs.shape) #(2, 1000)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
results = decode_predictions(probs) #1000개 중에 하나 선택해서 복구화
print("----------------------------------")
print("result[0] : ",results[0])
print("----------------------------------")
print("result[1] : ",results[1])
print("----------------------------------")
print("result[2] : ",results[2])
print("----------------------------------")
print("result[3] : ",results[3])


'''
result[0] :  [('n02100583', 'vizsla', 0.2956959), ('n02099849', 'Chesapeake_Bay_retriever', 0.28629395), ('n02092339', 'Weimaraner', 0.21879296), ('n02099712', 'Labrador_retriever', 0.14888147), ('n02099601', 'golden_retriever', 0.03657168)]
----------------------------------
result[1] :  [('n02123045', 'tabby', 0.39644343), ('n02123159', 'tiger_cat', 0.35422727), ('n02124075', 'Egyptian_cat', 0.17089514), ('n03223299', 'doormat', 0.011577507), ('n02123394', 'Persian_cat', 0.0056501427)]
----------------------------------
result[2] :  [('n03291819', 'envelope', 0.21738464), ('n02786058', 'Band_Aid', 0.09120862), ('n03598930', 'jigsaw_puzzle', 0.0648257), ('n03908618', 'pencil_box', 0.058071237), ('n06359193', 'web_site', 0.056942582)]
----------------------------------
result[3] :  [('n02906734', 'broom', 0.33535185), ('n04350905', 'suit', 0.2463068), ('n03141823', 'crutch', 0.084583715), ('n04367480', 'swab', 0.06147502), ('n03680355', 'Loafer', 0.05383112)]
'''