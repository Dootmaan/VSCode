#GMM with EM
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import copy

#生成数据
class0=40
class1=30
class2=50
class_num=3
total_num=class0+class1+class2
dimension=2
e=0.00005

curvedata=list()    

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X=np.random.randn(1,dimension)
    X[0,dimension-1]=X[0,dimension-1]+10
    X[0,0]=X[0,0]-5
    curvedata.append(X)

for i in range(class1):
    X=2*np.random.randn(1,dimension)+10
    curvedata.append(X)

for i in range(class2):
    X=3*np.random.randn(1,dimension)
    curvedata.append(X)

random.shuffle(curvedata)

def normDis(x,miu,sigma):
    # return (1.0/(pow((2*np.pi),dimension/2.0))*pow(np.linalg.det(sigma),0.5))*(np.exp(-(x-miu).T*np.linalg.inv(sigma)*((x-miu))/2))   #多维高斯分布
    return (1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp((-0.5) * (np.dot(np.dot((x-miu), np.linalg.inv(sigma)), (x-miu).T)))

miu_history=list()
    #初始值随便
miu=list()
miu.append(np.array([[0,10]]))
miu.append(np.array([[10,10]]))
miu.append(np.array([[1,1]]))  

miu_history.append(miu)      
          
sigma=list()
x=np.zeros(shape=(2,2))
x=[[1,0],[0,1]]
sigma.append(x)

x=np.zeros(shape=(2,2))
x=[[2,0],[0,2]]
sigma.append(x)

x=np.zeros(shape=(2,2))
x=[[3,0],[0,3]]
sigma.append(x)

k=[3,3,5]

def isChanged(a,b):
    for i in range(class_num):
        if(abs(a[i]-b[i])>e):
            return True
    return False

GAMA=np.ones(shape=(class_num,total_num))   #第i个样本落在第j类的概率（3*330）
Nk=[0,0,0]

def EM():
    while(True):   #迭代固定次数
        for i in range(class_num):
            for j in range(total_num):
                SUM=0
                for t in range(class_num):
                    SUM+=k[t]*normDis(curvedata[j],miu[t],sigma[t])
                GAMA[i,j] = (k[i]*normDis(curvedata[j],miu[i],sigma[i]))/SUM   #E步结束,第j个样本属于第i类的概率
        old_Nk=[0,0,0]
        for i in range(class_num):
            old_Nk[i]=Nk[i]
                # print(GAMA[i,j])
        for i in range(class_num):
            Nk[i]=np.sum(GAMA[i,:])  #M步完成，计算出有多少个样本属于i类
        print(Nk)
        #开始更新
        for i in range(class_num):
            k[i] = 1.0*Nk[i]/total_num
        for i in range(class_num): 
            temp=np.zeros(shape=(1,dimension))
            for j in range(dimension):
                s=0
                for q in range(total_num):
                    s+=GAMA[i,q]*curvedata[q][0,j]
                temp[0,j]=s/Nk[i]
            miu[i]=temp
            # miu[i] = 1.0*np.sum([GAMA[i,t]*curvedata[t] for t in range(total_num)])/Nk[i]
        miu_history.append(copy.deepcopy(miu))
        for i in range(class_num):
            s=0
            for j in range(total_num):
                s+=GAMA[i,j]*((curvedata[j]-miu[i]).T)*(curvedata[j]-miu[i])
            sigma[i] = s/Nk[i]
            # sigma[i] = np.sqrt(1.0*np.sum(GAMA[i,:]*(curvedata-miu[i])*(curvedata-miu[i]))/Nk[i])    #方差不要忘记开方！！  
        if(isChanged(old_Nk,Nk)==False):
            break

def Print():
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for j in range(total_num):
        c=0
        temp=0  
        for i in range(class_num):   #根据概率大小将样本点分类
            if(GAMA[i,j]>temp):
                temp=GAMA[i,j]
                c=i
        if(c==0):
                xcord0.append(curvedata[j][0,0])
                ycord0.append(curvedata[j][0,1])
        if(c==1):
                xcord1.append(curvedata[j][0,0])
                ycord1.append(curvedata[j][0,1])
        if(c==2):
                xcord2.append(curvedata[j][0,0])
                ycord2.append(curvedata[j][0,1])
    plt.scatter(xcord0,ycord0,s=30,c='blue')
    plt.scatter(xcord1, ycord1,s=30, c='red')
    plt.scatter(xcord2, ycord2, s=30, c='green')
    plt.show()

def Print1(a):
    x=list()
    y1=list()
    y2=list()
    for i in range(len(a)):
        x.append(i)
        y1.append(a[i][0,0])
        y2.append(a[i][0,1])
    plt.plot(x,y1,'g')
    plt.plot(x,y2,'r')
    plt.show()

def Print2(a):
    x=list()
    y1=list()
    y2=list()
    for i in range(len(a)):
        x.append(i)
        y1.append(a[i][0,0])
        y2.append(a[i][0,1])
    plt.plot(x,y1,'g')
    plt.plot(x,y2,'r')
    plt.show()

def Print3(a):
    x=list()
    y1=list()
    y2=list()
    for i in range(len(a)):
        x.append(i)
        y1.append(a[i][0,0])
        y2.append(a[i][0,1])
    plt.plot(x,y1,'g')
    plt.plot(x,y2,'r')
    plt.show()

def PrintTra():
    a=list()
    b=list()
    c=list()
    for miu in miu_history:
        a.append(miu[0])
        b.append(miu[1])
        c.append(miu[2])
    Print1(a)
    Print2(b)
    Print3(c)

EM()
print("预测结果如下")
print(k)
print(miu)
print(sigma)
Print()
PrintTra()