#GMM with EM
import numpy as np
import math
import matplotlib.pyplot as plt
import random

dimension=3
class_num=2
total_num=0
e=0.005
#生成数据
curvedata=list()  
pos_num=0
neg_num=0

lines=list()
try:
    f = open('C:\\Users\\samma\\Desktop\\uci.txt', 'r')
    lines=f.readlines()
    total_num=len(lines)
finally:
    if f:
        f.close()

for i in lines:
    temp=i.split(',')
    X=np.matrix([[int(temp[0]),int(temp[1]),int(temp[2])]])
    if(int(temp[3])==1):
        neg_num+=1
    if(int(temp[3])==2):
        pos_num+=1
    curvedata.append(X)

def normDis(x,miu,sigma):
    # return (1.0/(pow((2*np.pi),dimension/2.0))*pow(np.linalg.det(sigma),0.5))*(np.exp(-(x-miu).T*np.linalg.inv(sigma)*((x-miu))/2))   #多维高斯分布
    return (1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp((-0.5) * (np.dot(np.dot((x-miu), np.linalg.inv(sigma)), (x-miu).T)))

miu=list()
miu.append(np.matrix([[40,60,1]]))
miu.append(np.matrix([[30,60,0]]))
sigma=list()
sigma.append(np.matrix([[3,0,0],[0,1,0],[0,0,1]]))
sigma.append(np.matrix([[2,0,0],[0,2,0],[0,1,2]]))

k=[2,1]

def isChanged(a,b):
    for i in range(class_num):
        if(abs(a[i]-b[i])>e):
            return True
    return False

GAMA=np.ones(shape=(class_num,total_num))   #第i个样本落在第j类的概率（3*330）
Nk=[0,0]  #和维度有关

def EM():
    while(True):   #迭代固定次数
        for i in range(class_num):
            for j in range(total_num):
                SUM=0
                for t in range(class_num):
                    SUM+=k[t]*normDis(curvedata[j],miu[t],sigma[t])
                GAMA[i,j] = (k[i]*normDis(curvedata[j],miu[i],sigma[i]))/SUM   #E步结束,第j个样本属于第i类的概率
        old_Nk=[0,0]
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
        for i in range(class_num):
            s=0
            for j in range(total_num):
                s+=GAMA[i,j]*((curvedata[j]-miu[i]).T)*(curvedata[j]-miu[i])
            sigma[i] = s/Nk[i]
            # sigma[i] = np.sqrt(1.0*np.sum(GAMA[i,:]*(curvedata-miu[i])*(curvedata-miu[i]))/Nk[i])    #方差不要忘记开方！！  
        if(isChanged(old_Nk,Nk)==False):
            break

EM()   
print("预测结果如下")
print(k)
print(miu)
print(sigma)
print("实际两类占数量为")
print(neg_num,pos_num)