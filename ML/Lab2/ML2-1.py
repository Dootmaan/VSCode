#梯度下降
import numpy as np
import math
import matplotlib.pyplot as plt
import random
#W=sigma(X_i(Y-P(Y=1|X,W)))

#手工生成训练数据
pos_num=70
neg_num=30
number=pos_num+neg_num
step=2/number
ALPHA=0.005
dimension=2
e=0.005

testdata=list()
curvedata=list()

class Dot:
    def __init__(self,_x,_y):
        self.x=_x
        self.y=_y

for i in range(pos_num):
    Temp=3*np.random.randn(1,dimension) + 1    #~N(10,10)
    X=np.zeros(shape=(1,dimension+1))
    X[0,0]=1
    for j in range(dimension):
        X[0,j+1]=Temp[0,j]
    d=Dot(X,1)
    curvedata.append(d)
for i in range(neg_num):
    Temp=2*np.random.randn(1,dimension) +7  #~N(1,2)
    X=np.zeros(shape=(1,dimension+1))
    X[0,0]=1
    for j in range(dimension):
        X[0,j+1]=Temp[0,j]
    d=Dot(X,0)
    curvedata.append(d)

#w0=1
dataMatrix = np.zeros(shape=(number,dimension+1))   #个数行，维度+1列
labelMatrix = np.zeros(shape=(number,1))    #个数行，一列
for i in range(number):
    labelMatrix[i,0]=curvedata[i].y
    for j in range(dimension+1):
        dataMatrix[i,j]=(curvedata[i].x)[0,j]

def sigmoid(x):
    return 1/(1+np.exp(-x))

W=np.ones(shape=(dimension+1,1))   #W is a vector

def grad(W):
    h = sigmoid(dataMatrix.dot(W))
    W= W + ALPHA * (dataMatrix.T).dot(labelMatrix - h)    #维数+1行，一列
    return W

def ifcontinue(W):
    W2=W
    W=grad(W)
    flag=False
    for i in range(dimension+1):
        if(abs(W2[i,0]-W[i,0])>e):
            flag=True
            break
    return flag

while(ifcontinue(W)):
    W=grad(W)
W=grad(W)

print(W)

def plotAll(W):
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(number):
        if labelMatrix[i] == 1:
            xcord1.append(dataMatrix[i][1])
            ycord1.append(dataMatrix[i][2])
        else:
            xcord2.append(dataMatrix[i][1])
            ycord2.append(dataMatrix[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30, c='red', marker='s')  #正例
    ax.scatter(xcord2, ycord2, s=30, c='green')           #反例
    x = np.arange(-3, 3, 0.1)
    y = (-W[0, 0] - W[1, 0] * x) / W[2, 0]  #matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

plotAll(W)
