from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import heapq

total_number=1000
test_number=200

def distance(a,b):
    distance=0
    for i in range(28):
        for j in range(28):
            distance += abs(a[28*i+j]-b[28*i+j])   #不是欧式距
    return distance

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels

all_imgs = []   #其中包含了各个数字的图片列表
img_labels=[]
centers=[]

for i in range(10):
    centers.append(0)  #初始化模板-也即中心点
    
    part_img=[]
    for j in range(total_number):
        if labels[j]==i:
            part_img.append(imgs[j])
    all_imgs.append(part_img)

def classify(img):
    min_dis=distance(img,centers[0])
    min_index=0
    for i in range(1,10):
        dis=distance(img,centers[i])
        if dis<min_dis:
            min_dis=dis
            min_index=i
    return min_index

def center():
    for i in range(10):
        s=0
        for img in all_imgs[i]:    #第i类中的所有图片
            s+=img
        centers[i]=s/float(len(all_imgs[i]))   #不能为0

results=[]
center()
for i in range(total_number,total_number+test_number):
    results.append(classify(imgs[i]))

print(results)

right_num=0
index=total_number
for result in results:
    if result == labels[index]:
        right_num+=1
    index+=1

print(float(right_num/test_number))
