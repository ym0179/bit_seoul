#Day22
#2020-12-08

import p11_car
import p12_tv

'''
운전하다.
car.py의 module 이름은  p11_car
시청하다.
tv.py의 module 이름은  p12_tv
'''

print("=======================================")
print("do.py의 module 이름은 ", __name__) #__main__
print("=======================================")

p11_car.drive()
p12_tv.watch()