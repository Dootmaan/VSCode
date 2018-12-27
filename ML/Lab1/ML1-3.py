#y=sin(2pi*x) 
# 梯度下降
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#数据个数，最好偶数
number=50
step=1/number
halfnumber=25  #number的一半

#控制次数
degree=100
LAMBDA=0.00005
GAMMA=0.001  #梯度下降的系数
e=0.000001 #停止梯度下降的边界

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

#print(curvedata[1].x)
X=np.zeros(shape=(halfnumber,degree+1))
T=np.zeros(shape=(halfnumber,1))
for i in range(halfnumber):
    T[i]=curvedata[i].y
    for j in range(degree+1):   #第一列是常数
        X[i,j]=math.pow(curvedata[i].x,j)
#print(T)

def J(W):
    return ((np.dot(X,W)-T).T).dot(np.dot(X,W)-T)+LAMBDA*(np.dot(W.T,W))[0,0]    #结果是数字？
def grad(W):
    return 2*np.dot(X.T,X).dot(W)-2*(X.T.dot(T))+2*(LAMBDA*(W))   #结果是(LAMBDA+1,1)矩阵

W=np.zeros(shape=(degree+1,1))  #W是最终结果
def ifcontinue(W):
    W2=W-GAMMA*grad(W)
    if abs(J(W2)-J(W))>e:
        return True
    else:
        return False
while(ifcontinue(W)):
    W=W-GAMMA*grad(W)
W=W-GAMMA*grad(W)  #因为判断中满足条件的那一次运算在循环体中未执行

print("系数矩阵为：",W)

x=np.zeros(shape=(halfnumber,1))
for i in range(1,halfnumber):
    x[i]=i*2*step+step

accuracy=0     #测试生成曲线的准确性
for dot in testdata:
    temp=np.zeros(shape=(1,degree+1))
    for j in range(degree+1):
        temp[0,j]=math.pow(dot.x,j)
    accuracy+=math.pow((dot.y-temp.dot(W))[0,0],2)
print(accuracy)

plt.plot(x,X.dot(W),'r') #拟合曲线
plt.plot(x,T,'g.') #自己生成的数据
plt.title("degree="+str(degree))
plt.show()