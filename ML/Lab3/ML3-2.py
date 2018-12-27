#GMM with EM
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#生成数据
class0=400
class1=300
class2=500
class_num=3
total_num=class0+class1+class2
tru_sigma=[2,4,3]
tru_miu=[10,30,60]

curvedata=list()    

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X=np.random.normal(tru_miu[0],tru_sigma[0])
    curvedata.append(X)

for i in range(class1):
    X=np.random.normal(tru_miu[1],tru_sigma[1])
    curvedata.append(X)

for i in range(class2):
    X=np.random.normal(tru_miu[2],tru_sigma[2])
    curvedata.append(X)

random.shuffle(curvedata)

def normDis(x,miu,sigma):
    return (1.0/np.sqrt(2*np.pi))*(np.exp(-pow((x-miu),2)/(2*pow(sigma,2))))   #高斯分布

k=[4,3,5]    #初始值随便
miu=[10,25,48]
sigma=[1,4,3]

times=20  #迭代次数
GAMA=np.ones(shape=(class_num,total_num))   #第i个样本落在第j类的概率（3*330）

def EM():
    for whatever in range(times):   #迭代固定次数
        for i in range(class_num):
            for j in range(total_num):
                SUM=0
                for t in range(class_num):
                    SUM+=k[t]*normDis(np.sum(curvedata[j]),miu[t],sigma[t])
                GAMA[i,j] = (k[i]*normDis(np.sum(curvedata[j]),miu[i],sigma[i]))/SUM   #E步结束,第j个样本属于第i类的概率
                # print(GAMA[i,j])
        Nk=[0,0,0]
        for i in range(class_num):
            Nk[i]=np.sum(GAMA[i,:])  #M步完成，计算出有多少个样本属于i类
        print(Nk)
        #开始更新
        for i in range(class_num):
            k[i] = 1.0*Nk[i]/total_num
        for i in range(class_num): 
            miu[i] = 1.0*np.sum(GAMA[i,:]*curvedata)/Nk[i]
        for i in range(class_num):
            sigma[i] = np.sqrt(1.0*np.sum(GAMA[i,:]*(curvedata-miu[i])*(curvedata-miu[i]))/Nk[i])    #方差不要忘记开方！！  

EM()   
print("预测结果如下")
print(k)
print(miu)
print(sigma)