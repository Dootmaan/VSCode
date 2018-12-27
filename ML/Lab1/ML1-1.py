#y=sin(2pi*x) 
# 求导，导数为0直接得解析解
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#数据个数，最好偶数
number=50
step=1/number
halfnumber=25  #number的一半，当然其实也无所谓

#控制次数
degree=25

testdata=list()
curvedata=list()

class Dot:
    def __init__(self,_x,_y):
        self.x=_x
        self.y=_y

#生成点数据
for i in range(number):
    dot=Dot(i*step,math.sin(2*math.pi*i*step)+(random.random()-0.5)/20)
    if i%2==0:
        testdata.append(dot)
    else:
        curvedata.append(dot)

print(curvedata[1].x)
X=np.zeros(shape=(halfnumber,degree+1))
T=np.zeros(shape=(halfnumber,1))
for i in range(halfnumber):
    T[i]=curvedata[i].y
    for j in range(degree+1):   #第一列是常数
        X[i,j]=math.pow(curvedata[i].x,j)
#print(T)

A=np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(T)
print("系数矩阵为：",X.dot(A))

x=np.zeros(shape=(halfnumber,1))
for i in range(1,halfnumber):
    x[i]=i*2*step+step

accuracy=0     #测试生成曲线的准确性
for dot in testdata:
    temp=np.zeros(shape=(1,degree+1))
    for j in range(degree+1):
        temp[0,j]=math.pow(dot.x,j)
    accuracy+=math.pow((dot.y-temp.dot(A))[0,0],2)
print(accuracy)

plt.plot(x,X.dot(A),'r') #拟合曲线
plt.plot(x,T,'g.') #自己生成的数据
plt.title("degree="+str(degree))
plt.show()

