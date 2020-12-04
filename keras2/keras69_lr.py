#Day18
#2020-12-02

weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 0.0001 #0.01, 0.1, 1 , 0.001, , 0.0001, 10
# 너무 작게 주면 range 안에서 0.8까지 안가고
# 너무 크게 주면 튕겨서 나감

for iteration in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) **2
    
    print("Error : " + str(error) + "\tPrediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr