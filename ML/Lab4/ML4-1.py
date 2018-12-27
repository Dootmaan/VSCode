#PCA
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import heapq
from mpl_toolkits.mplot3d import axes3d

#生成数据
class0=50
class1=120
class2=110
class_num=3
total_num=class0+class1+class2
dimension=3  #数据特征维度
zipped_dimension=2

curvedata=list()
data=np.zeros(shape=(total_num,dimension))   

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X=3*np.random.randn(1,dimension)+10
    for j in range(dimension-zipped_dimension+1,1,-1):
        X[0,j]=2*X[0,j-1]
    curvedata.append(X)

for i in range(class1):
    X=4*np.random.randn(1,dimension)+20
    for j in range(dimension-zipped_dimension+1,1,-1):
        X[0,j]=3*X[0,j-1]
    curvedata.append(X)

for i in range(class2):
    X=2*np.random.randn(1,dimension)+30
    for j in range(dimension-zipped_dimension+1,1,-1):
        X[0,j]=4*X[0,j-1]
    curvedata.append(X)

for i in range(total_num):
    data[i,:]=curvedata[i]

def pca():
    #求数据矩阵每一列的均值
    mean=list()
    for i in range(dimension):
        s=0
        for j in range(total_num):
            s+=data[j,i]
        avg=s*1.0/total_num
        mean.append(avg)
    meanMatrix=np.zeros(shape=(1,dimension))
    for i in range(dimension):
        meanMatrix[0,i]=mean[i]
    #均值大家都有，全删掉再说
    meanRemoved=data-meanMatrix
    covMat=np.cov(meanRemoved,rowvar=False)  #rowvar指明每一列是一个变量
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))   #eig方法返回特征值和特征向量

    nmaxColumn=heapq.nlargest(zipped_dimension, range(len(eigVals)), eigVals.take)   #找到最大的几个特征值对应的下标
    # print(nmaxColumn)

    zipped_feature=eigVects[:,nmaxColumn]   #用上面找到的下标提取最大的几个特征向量,zipped_feature显然是dimension
    zipped_data=meanRemoved.dot(zipped_feature)  #投影

    #反构出原数据矩阵
    reconMat=(zipped_data.dot(zipped_feature.T))+meanMatrix
    # print(reconMat)
    
    #返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return zipped_data,reconMat

def PrintZip(a):
    xcord = []
    ycord = []
    for j in range(total_num):
        xcord.append(a[j,0])
        ycord.append(a[j,1])
    plt.scatter(xcord,ycord,s=30,c='blue')
    plt.show()
def PrintBack(b):
    xcord = []
    ycord = []
    zcord=[]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for j in range(total_num):
        xcord.append(b[j,0])
        ycord.append(b[j,1])
        zcord.append(b[j,2])
    ax.scatter(xcord,ycord,zcord,c='blue')
    plt.show()
def PrintRaw(raw):
    xcord = []
    ycord = []
    zcord=[]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for j in range(total_num):
        xcord.append(raw[j,0])
        ycord.append(raw[j,1])
        zcord.append(raw[j,2])
    ax.scatter(xcord,ycord,zcord,c='blue')
    plt.show()

a,b=pca()
print("降维后：")
print(a)
print("还原后：")
print(b)
PrintZip(a)
PrintBack(b)
PrintRaw(data)