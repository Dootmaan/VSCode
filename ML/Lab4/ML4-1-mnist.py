from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import heapq

dimension=784
total_num=100
zipped_dimension=1  #设置要压缩成几维
select_num=4  #要选择选取的mnist中的数字

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels

origin_imgs = []
for i in range(10000):
      if labels[i] == select_num and len(origin_imgs) < 100:
          origin_imgs.append(imgs[i])

def array_to_img(array):
    array=array*255
    new_img=Image.fromarray(array.astype(np.uint8))
    return new_img

def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
     new_img = Image.new(new_type, (col* each_width, row* each_height)) 
     for i in range(len(origin_imgs)):
         each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_width))
         new_img.paste(each_img, ((i % col) * each_width, (int)(i / col) * each_width)) 
     return new_img

#降维度前
ten_origin_imgs=comb_imgs(origin_imgs, 10, 10, 28, 28, 'L')
ten_origin_imgs.show()

def pca(data):
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

a,b=pca(np.array(origin_imgs))
low_d_img = comb_imgs(b, 10, 10, 28, 28, 'L')
low_d_img.show()
print(b)
