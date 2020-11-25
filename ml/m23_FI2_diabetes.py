#Day13
#2020-11-25

#xgboost
#1. FI 0 제거 또는 2. 하위 30% 제거
#3. 디폴트랑 성능 비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

x,y = load_diabetes(return_X_y=True)

x = x[:,[1,2,3,5,6,7,8,9]]

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
print(x_train.shape) #(353, 10)

model = XGBRegressor(max_depth=4)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ", acc)

print(model.feature_importances_)
# [0.03951401 0.08722725 0.18159387 0.08551976 0.04845208 0.06130722
#  0.05748899 0.0561045  0.32311246 0.05967987]

def get_drop_list(n):
    a = model.feature_importances_.tolist()
    b = a.copy()
    b.sort()
    b = b[:n] #앞에 몇개
    index_list = []
    for i in model.feature_importances_:
        if i in b:
            index_list.append(a.index(i))
    return(index_list)

drop_list = get_drop_list(3)
print(drop_list)

'''
default
acc :  0.31163770597265394

FI 제일 하위 column (0,4)번째 제거
acc :  0.3334780579480626
'''