import cv2 as cv2
import sys
import math
import struct
import matplotlib.pyplot as plt
import numpy as np
import threading
import os.path


def changeContrast(img, contrast):
    contrast = float(contrast)
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            tmp = 127.5 + contrast * (int(img[i][j][0]) - 127.5)
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][0] = tmp
    for i in range(row):
        for j in range(col):
            tmp = 127.5 + contrast * (int(img[i][j][1]) - 127.5)
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][1] = tmp
    for i in range(row):
        for j in range(col):
            tmp = 127.5 + contrast * (int(img[i][j][2]) - 127.5)
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][2] = tmp


def changeBrightness(img, brightness):
    brightness = float(brightness)
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            tmp = brightness * int(img[i][j][0])
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][0] = tmp
    for i in range(row):
        for j in range(col):
            tmp = brightness * int(img[i][j][1])
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][1] = tmp
    for i in range(row):
        for j in range(col):
            tmp = brightness * int(img[i][j][2])
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            img[i][j][2] = tmp


def avgSmoothing(img, m=3, n=3):
    row = img.shape[0]
    col = img.shape[1]
    for i in range(int(m / 2), row - int(m / 2)):
        for j in range(int(n / 2), col - int(n / 2)):
            for k in range(3):
                summary = int(0)
                for a in range(-int(m / 2), int(m / 2) + 1):
                    for b in range(-int(n / 2), int(n / 2) + 1):
                        if (a == 0 and b == 0):
                            continue
                        summary += int(img[i + a][j + b][k])
                avg = summary / (m * n - 1)
                if (abs(img[i][j][k] - avg) > 5):
                    img[i][j][k] = avg


def medianSmoothing_quic(img, m=3, n=3):
    row = img.shape[0]
    col = img.shape[1]
    new_img = img.copy()
    th = (m * n) / 2 + 1
    for k in range(3):
        for i in range(int(m / 2), row - int(m / 2)):
            hist = []
            for whatever in range(256):
                hist.append(0)
            temp = []
            for a in range(-int(m / 2), int(m / 2) + 1):
                for b in range(n):
                    temp.append(img[i + a][b][k])
                    hist[img[i + a][b][k]] += 1
            median = np.median(temp)
            new_img[i][int(n / 2)][k] = median
            ltmdn = 0
            for t in temp:
                if (t <= median):
                    ltmdn += 1
            for j in range(int(n / 2) + 1, col - int(n / 2)):
                for a in range(-int(m / 2), int(m / 2) + 1):
                    tmp = img[i + a][j - int(n / 2) - 1][k]
                    hist[tmp] -= 1
                    if (tmp <= median):
                        ltmdn -= 1
                    tmp = img[i + a][j + int(n / 2)][k]
                    hist[tmp] += 1
                    if (tmp <= median):
                        ltmdn += 1
                if (ltmdn >= th):
                    while (ltmdn - hist[int(median)] >= th
                           or hist[int(median)] == 0 or ltmdn < th):
                        ltmdn -= hist[int(median)]
                        median -= 1
                else:
                    while (ltmdn - hist[int(median)] >= th
                           or hist[int(median)] == 0 or ltmdn < th):
                        median += 1
                        ltmdn += hist[int(median)]
                new_img[i][j][k] = median
    B, G, R = cv2.split(new_img)
    img[:, :, 0] = B
    img[:, :, 1] = G
    img[:, :, 2] = R


def sobelOutline(img, level=90):
    row = img.shape[0]
    col = img.shape[1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outline_img = np.zeros((row, col), np.uint8)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            sobel = abs(
                int(gray_img[i - 1][j - 1]) + 2 * int(gray_img[i - 1][j]) +
                int(gray_img[i - 1][j + 1]) - int(gray_img[i + 1][j + 1]) -
                2 * int(gray_img[i + 1][j]) - int(gray_img[i + 1][j - 1]))
            sobel += abs(
                int(gray_img[i - 1][j - 1]) + 2 * int(gray_img[i][j - 1]) +
                int(gray_img[i + 1][j - 1]) - int(gray_img[i - 1][j + 1]) -
                2 * int(gray_img[i][j + 1]) - int(gray_img[i + 1][j + 1]))
            if (sobel > level):
                outline_img[i][j] = 255
    return outline_img


def show(src, img):
    cv2.imshow(src, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def export(dest, img):
    cv2.imwrite(dest, img)
    print("Successfully export to " + dest)


'''
Now the Lab3 main section.
'''


def flood(img):
    row = img.shape[0]
    col = img.shape[1]
    cv2.floodFill(img, np.zeros([row + 2, col + 2], np.uint8), (900, 200),
                  (255, 255, 255), (80, 10, 10), (80, 10, 10), 0)

    cv2.floodFill(img, np.zeros([row + 2, col + 2],
                                np.uint8), (900, 700), (255, 255, 255),
                  (80, 80, 80), (80, 80, 80), cv2.FLOODFILL_FIXED_RANGE)

    cv2.floodFill(img, np.zeros([row + 2, col + 2],
                                np.uint8), (300, 700), (255, 255, 255),
                  (90, 90, 90), (90, 90, 90), cv2.FLOODFILL_FIXED_RANGE)


def rmBg(img, level=100, m=5, n=5):
    origin = img.copy()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # show('temp', outline)
    row = img.shape[0]
    col = img.shape[1]

    #背景参考像素点
    standard = hsv_img[50, 50, :]
    for i in range(int(m / 2), row - int(m / 2)):
        for j in range(int(n / 2), col - int(n / 2)):
            s = 0
            max = 0
            for k in range(3):
                tmp = abs(int(hsv_img[i, j, k]) - int(standard[k]))
                s += tmp
                if (tmp > max):
                    max = tmp
            if (s < 3 * level and max < 180):  #相似像素，且通道最大差满足条件置白色
                img[i, j, :] = 255

    # 下面可以尝试对初步提取出的图片做边缘检测找到轮廓填充内容！！！！！
    outline = sobelOutline(img, level=50)
    for i in range(row):
        for j in range(col):
            for k in range(3):
                tmp = abs(int(hsv_img[i, j, k]) - int(standard[k]))
                s += tmp
                if (tmp > max):
                    max = tmp
            if (outline[i][j] == 0 and s < 4 * level
                    and max < 200):  #进一步去背景，放宽限制
                img[i, j, :] = 255
            elif (outline[i][j] == 255
                  and img[i][j][:].any == 255):  #补全过渡去掉的图像边缘
                img[i, j, :] = origin[i, j, :]


#已知水印所在位置为340，842~680,842
def rmWatermark(img):
    row = img.shape[0]
    col = img.shape[1]
    count = 1
    for i in range(row - 1, 842, -1):
        count += 1
        for j in range(340, 680):
            img[i, j, :] = img[i - row + 842 - count, j, :]

    for i in range(row - 1, 842, -1):
        alpha = 0.9
        for j in range(340, 360):
            img[i, j, :] = (1 - alpha) * img[i, j, :] + alpha * img[i, 339, :]
            alpha -= 0.04


'''
@param m/n 指明最后进行涂抹时的窗口大小 
'''


def rmFleck(img, m=3, n=3):
    # medianSmoothing_quic(img, m=5, n=5)  #先去噪声再提边缘，实际上可以注释掉然后开启下面两个
    changeContrast(img, 0.5)  #防止雀斑被边界检测太多
    outline = sobelOutline(img)
    changeContrast(img, 2)  #变回去
    row = img.shape[0]
    col = img.shape[1]

    #先中值去掉大部分雀斑，然后均值淡化剩下的雀斑
    changeContrast(img, 1.3)  #加深雀斑方便去噪
    raw_img = img.copy()
    medianSmoothing_quic(raw_img, m=5, n=5)
    avgSmoothing(raw_img)

    #强化边缘
    for i in range(int(m / 2), row - int(m / 2)):
        for j in range(int(n / 2), col - int(n / 2)):
            if (outline[i][j] == 255):
                for k in range(3):
                    raw_img[i][j][k] = img[i][j][k]

    img[:, :, 0] = raw_img[:, :, 0]
    img[:, :, 1] = raw_img[:, :, 1]
    img[:, :, 2] = raw_img[:, :, 2]

    #可选附加中值操作，实际提升不大
    medianSmoothing_quic(img)

    #涂抹，最后再次涂抹已经淡化的雀斑
    for i in range(int(m / 2), row - int(m / 2)):
        for j in range(int(n / 2), col - int(n / 2)):
            for k in range(3):
                summary = int(0)
                for a in range(-int(m / 2), int(m / 2) + 1):
                    for b in range(-int(n / 2), int(n / 2) + 1):
                        if (a == 0 and b == 0):
                            continue
                        summary += int(img[i + a][j + b][k])
                avg = summary / (m * n - 1)
                if (abs(img[i][j][k] - avg) < 5):
                    img[i][j][k] = avg

    #调节图象参数到正常值
    changeContrast(img, 0.8)
    changeBrightness(img, 1.1)


'''
@param alpha 透明度
'''


def beautify(img, alpha=50):
    row = img.shape[0]
    col = img.shape[1]
    target_img = img.copy()
    bi_img = cv2.bilateralFilter(target_img, 0, 100, 15)
    for i in range(row):
        for j in range(col):
            for k in range(3):
                tmp = int(bi_img[i][j][k]) - int(img[i][j][k]) + 128
                if (tmp > 255):
                    target_img[i][j][k] = 255
                elif (tmp < 0):
                    target_img[i][j][k] = 0
                else:
                    target_img[i][j][k] = tmp
    gu_img = cv2.GaussianBlur(target_img, (3, 3), 1)
    for i in range(row):
        for j in range(col):
            for k in range(3):
                tmp = (
                    (100 - alpha) * int(img[i][j][k]) + alpha *
                    (int(img[i][j][k]) + 2 * int(gu_img[i][j][k]) - 256)) / 100
                if (tmp > 255):
                    target_img[i][j][k] = 255
                elif (tmp < 0):
                    target_img[i][j][k] = 0
                else:
                    target_img[i][j][k] = tmp

    rmFleck(target_img)
    img[:, :, 0] = target_img[:, :, 0]
    img[:, :, 1] = target_img[:, :, 1]
    img[:, :, 2] = target_img[:, :, 2]


while (True):
    filesrc = input("Type in .bmp file src: ")
    if (os.path.isfile(filesrc)):
        rgb_img = cv2.imread(filesrc)
        break

while (True):
    print("=============\n")
    choice = input(
        "Choose the function: \n0.Preview\n1.Remove background\n2.Remove watermark\n3.Remove fleck\n4.Beautify all-roundly\n5.Export and Quit\n"
    )
    if (choice == '0'):
        print("Press any key to close.")
        t = threading.Thread(target=show, args=(filesrc, rgb_img))
        t.setDaemon(True)
        t.start()
    elif (choice == '1'):
        rmBg(rgb_img)
    elif (choice == '2'):
        rmWatermark(rgb_img)
    elif (choice == '3'):
        rmFleck(rgb_img)
    elif (choice == '4'):
        beautify(rgb_img)
    elif (choice == 'x'):
        print("Surprise!")
        flood(rgb_img)
    elif (choice == '5'):
        dest = input("Type in the dest path: ")
        export(dest, rgb_img)
        break
