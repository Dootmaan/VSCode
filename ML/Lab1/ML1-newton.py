#y=sin(2pi*x) 
# 牛顿法，未完成
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#共轭梯度本质是求解 W^T*X^T*X*W-2*W^TX^T*T 的最小值
#根据推导过程，最小值已经确定，只要找到共轭向量带入即可

#数据个数，最好偶数
number=50
step=1/number
halfnumber=25  #number的一半

#控制次数
degree=10
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

W=np.zeros(shape=(degree+1,1))  #W是最终结果

NORMAL=LAMBDA*np.ones(shape=[degree+1,degree+1]) #正则项
R=(X.T).dot(T)-((X.T).dot(X)+NORMAL).dot(W)
D=R

def ifbreak(R):
    ifbreak=True
    for i in range(degree+1):
        if R[i,0]>e:
            ifbreak=False
    return ifbreak

while(True):
    A=(R.T).dot(R).dot(np.linalg.inv((D.T).dot((X.T).dot(X)+NORMAL).dot(D)))
    W=W+A*D
    R_2=R-A*((X.T).dot(X)+NORMAL).dot(D)
    if(ifbreak(R_2)): 
        break
    B=(R_2.T).dot(R_2).dot(np.linalg.inv((R.T).dot(R)))
    D=R_2+B*D
    R=R_2

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