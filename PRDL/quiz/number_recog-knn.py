from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import heapq
import math

total_number = 1000  #训练集大小
test_number = 100  #测试集大小
k = 10


def distance(a, b):
    distance = 0
    for i in range(28):
        for j in range(28):
            distance += abs(a[28 * i + j] - b[28 * i + j])  #不是欧式距(这是啥距离来着？)
            # distance += math.sqrt((a[28*i+j]-b[28*i+j])**2)   #欧式距离，不仅慢而且没啥区别
    return distance


def classify(img):
    all_distances = []
    indexs = []
    for t_img in trainning_set:
        all_distances.append(distance(img, t_img))
    indexs = map(all_distances.index, heapq.nsmallest(k, all_distances))
    indexs = list(indexs)  #最小的k个点所对应的index，只需要查看labels[index]即可知道类别

    catagory = 0
    max_count = 0
    for i in range(10):
        s = 0
        for index in indexs:
            if labels[index] == i:
                s += 1
        if s > max_count:
            max_count = s
            catagory = i

    return catagory


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels

trainning_set = imgs[0:total_number]

results = []
for i in range(total_number, total_number + test_number):
    results.append(classify(imgs[i]))

# show_centers()     #图片展示centers中的模版图片

print(results)

#统计正确率
right_num = 0
index = total_number
for result in results:
    if result == labels[index]:
        right_num += 1
    index += 1

print('正确率： ' + str(float(right_num / test_number)))
