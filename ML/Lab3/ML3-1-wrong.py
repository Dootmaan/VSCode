#K-means
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#生成数据
class0=20
class1=30
class2=25
class_num=3
total_num=class0+class1+class2
dimension=2  #数据特征维度

curvedata=list()

class Dot:               #该类用来标识每个节点的类别
    def __init__(self,_x,_y):
        self.x=_x
        self.y=_y           

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X=1.5*np.random.randn(1,dimension)
    X[0,dimension-1]=X[0,dimension-1]+20
    X[0,0]=X[0,0]-5
    curvedata.append(X)

for i in range(class1):
    X=2*np.random.randn(1,dimension)+10
    curvedata.append(X)

for i in range(class2):
    X=np.random.randn(1,dimension)
    curvedata.append(X)

result=list()

#计算欧氏距
def distance(A, B):
    return math.sqrt(sum(pow(A[0] - B[0], 2)))

def re_center(k):
    myclass=list()
    for i in range(total_num):
        if(result[i].y==k):
            myclass.append(result[i].x)
    min=myclass[0]
    minsum=10000000000000
    length=len(myclass)
    sum=0
    #计算中心
    for i in range(length):
        for j in range(length):
            sum+=distance(myclass[j],myclass[i])
        if(sum<minsum):
            min=myclass[i]
    return min

def findIndex(l,data):
    length=len(l)
    for i in range(length):
        if((l[i]==data).all()):    #找到对应下标
            return i
    return -1

def contains(l,m):
    length=len(l)
    for i in range(length):
        if((l[i]==m).all()):
            return True
    return False

init_center = random.sample(curvedata, class_num)
for i in range(total_num):
    if(contains(init_center,curvedata[i])):
        result.append(Dot(curvedata[i],findIndex(init_center,curvedata[i])))
    else:
        result.append(Dot(curvedata[i],-1)) #所有点全初始化为-1类

for i in range(total_num):
    if(contains(init_center,curvedata[i])):
        continue
    else:
        dis=list()
        for j in range(class_num):
            dis.append(distance(init_center[j],curvedata[i]))   #计算该点与每一个中心的距离，存在dis中
        min=0
        for k in range(class_num):
            if(dis[k]<dis[min]):
                min=k      #找到最小距离对应的类别
        result[i].y=min   #第i点被设为k类
        init_center[min]=re_center(min)

#现在result中存放了分类结果，可以打印了
for i in range(total_num):
    print(result[i].y)   #高维度样本测试

def Print():
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(total_num):
        if(result[i].y == 0):
            xcord0.append(result[i].x[0,0])
            ycord0.append(result[i].x[0,1])
        elif(result[i].y == 1):
            xcord1.append(result[i].x[0,0])
            ycord1.append(result[i].x[0,1])
        else:
            xcord2.append(result[i].x[0,0])
            ycord2.append(result[i].x[0,1])
    plt.scatter(xcord0,ycord0,s=30,c='blue')
    # plt.scatter(init_center[0][0,0],init_center[0][0,1],c='blue',marker='s')
    plt.scatter(xcord1, ycord1,s=30, c='red') 
    # plt.scatter(init_center[0][0,0],init_center[0][0,1],c='blue',marker='s')
    plt.scatter(xcord2, ycord2, s=30, c='green') 
    # plt.scatter(init_center[0][0,0],init_center[0][0,1],c='blue',marker='s')
    plt.show()
    
Print()