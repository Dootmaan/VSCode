#K-means
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#生成数据
class0=100
class1=120
class2=110
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
    X=np.random.randn(1,dimension)
    X[0,dimension-1]=X[0,dimension-1]+10
    X[0,0]=X[0,0]-5
    curvedata.append(X)

for i in range(class1):
    X=1.2*np.random.randn(1,dimension)+10
    curvedata.append(X)

for i in range(class2):
    X=0.8*np.random.randn(1,dimension)
    curvedata.append(X)

result=list()

#计算欧氏距
def distance(A, B):
    return math.sqrt(sum(pow(A[0] - B[0], 2)))

def re_center():
    init_center2=list()
    for k in range(class_num):
        myclass=list()
        for i in range(total_num):
            if(result[i].y==k):
                myclass.append(result[i].x)
        length=len(myclass)
        axis=list()
        #计算中心
        for i in range(dimension):
            s=0
            for j in range(length):
                s+=myclass[j][0,i]
            axis.append(s/length)
        center=np.zeros(shape=(1,dimension))
        for i in range(dimension):
            center[0,i]=axis[i]
        init_center2.append(center)
    return init_center2

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

def equals(a,b):
    for i in range(total_num):
        if(a[i].y!=b[i]):    #局限性很强的函数，纯粹为了判断推退出条件写的
            return False
    return True

init_center =list()   #找class_num个间隔较远的初始点
First= random.sample(curvedata, 1)
init_center.append(First[0])
for i in range(class_num):
    max_dis=0
    max=init_center[0]
    for j in range(total_num):
        s=0
        length=len(init_center)
        for k in range(length):   #算最远距离
            dis=distance(curvedata[j],init_center[k])
            s+=dis
        if(s>max_dis):
            max_dis=s
            max=curvedata[j]
    init_center.append(max)
        
for i in range(total_num):
    if(contains(init_center,curvedata[i])):
        result.append(Dot(curvedata[i],findIndex(init_center,curvedata[i])))
    else:
        result.append(Dot(curvedata[i],-1)) #所有点全初始化为-1类

while(True):
    # if(contains(init_center,curvedata[i])):
    #     continue
    # else:
    flag=1
    result2=list()
    for i in range(total_num):
        result2.append(result[i].y)
        dis=list()
        for j in range(class_num):
            dis.append(distance(init_center[j],curvedata[i]))   #计算该点与每一个中心的距离，存在dis中
        min=0
        for k in range(class_num):
            if(dis[k]<dis[min]):
                min=k      #找到最小距离对应的类别
        result[i].y=min   #第i点被设为k类
    if(equals(result,result2)):
        break
    else:
        init_center=re_center()
    

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
    plt.scatter(init_center[0][0,0],init_center[0][0,1],c='blue',marker='s')
    plt.scatter(xcord1, ycord1,s=30, c='red') 
    plt.scatter(init_center[1][0,0],init_center[1][0,1],c='red',marker='s')
    plt.scatter(xcord2, ycord2, s=30, c='green') 
    plt.scatter(init_center[2][0,0],init_center[2][0,1],c='green',marker='s')
    plt.show()
    
Print()
