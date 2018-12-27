import cv2 as cv2
import sys
import math
import struct
import matplotlib.pyplot as plt
import numpy as np
import threading
import os.path


def changeContrast(img):
    contrast = input("Type in contrast: ")
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


def changeBrightness(img):
    brightness = input("Type in brightness: ")
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


def changeSaturation(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = input("Type in saturation: ")
    saturation = float(saturation)
    row = img.shape[0]
    col = img.shape[1]
    H, S, V = cv2.split(temp)
    for i in range(row):
        for j in range(col):
            tmp = saturation * temp[i][j][1]
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            temp[i][j][1] = tmp

    temp2 = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    B2, G2, R2 = cv2.split(temp2)

    img[:, :, 0] = B2
    img[:, :, 1] = G2
    img[:, :, 2] = R2


def changeHue(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = input("Type in hue: ")
    hue = float(hue)
    row = img.shape[0]
    col = img.shape[1]
    H, S, V = cv2.split(temp)
    for i in range(row):
        for j in range(col):
            tmp = hue + temp[i][j][0]
            if (tmp < 0):
                tmp = 0
            if (tmp > 255):
                tmp = 255
            temp[i][j][0] = tmp

    temp2 = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    B2, G2, R2 = cv2.split(temp2)

    img[:, :, 0] = B2
    img[:, :, 1] = G2
    img[:, :, 2] = R2


def getHistogram(img):
    row = img.shape[0]
    col = img.shape[1]
    total = row * col
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = []
    x = []
    for i in range(256):
        hist.append(float(0))
        x.append(i)
    for i in range(row):
        for j in range(col):
            hist[gray_img[i][j]] += 1 / total

    plt.bar(x, hist, label="Gray_img Histogram")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gray_Histogram')
    plt.show()


def avgSmoothing(img, m=5, n=5):
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


def medianSmoothing(img, m=3, n=3):
    row = img.shape[0]
    col = img.shape[1]
    for i in range(int(m / 2), row - int(m / 2)):
        for j in range(int(n / 2), col - int(n / 2)):
            for k in range(3):
                temp = []
                for a in range(-int(m / 2), int(m / 2) + 1):
                    for b in range(-int(n / 2), int(n / 2) + 1):
                        temp.append(img[i + a][j + b][k])
                median = np.median(temp)
                img[i][j][k] = median


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


def robertOutline(img, level=20):
    row = img.shape[0]
    col = img.shape[1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outline_img = np.zeros((row, col, 3), np.uint8)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            robert = max(
                abs(int(gray_img[i - 1][j - 1]) - int(gray_img[i + 1][j + 1])),
                abs(int(gray_img[i - 1][j + 1]) - int(gray_img[i + 1][j - 1])))
            if (abs(gray_img[i][j] - robert) > level):
                outline_img[i][j] = robert
    cv2.imwrite('D:\\robert.bmp', outline_img)
    print("Pic saved to D:\\robert.bmp")


def sobelOutline(img, level=150):
    row = img.shape[0]
    col = img.shape[1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outline_img = np.zeros((row, col, 3), np.uint8)
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
                outline_img[i][j] = sobel
    cv2.imwrite('D:\\sobel.bmp', outline_img)
    print("Pic saved to D:\\sobel.bmp")


def show(src, img):
    cv2.imshow(src, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows


def export(dest, img):
    cv2.imwrite(dest, img)
    print("Successfully export to " + dest)


while (True):
    filesrc = input("Type in .bmp file src: ")
    if (os.path.isfile(filesrc)):
        rgb_img = cv2.imread(filesrc)
        break

while (True):
    print("=============\n")
    choice = input(
        "Choose the function: \n0.Preview\n1.Change contrast\n2.Change brightness\n3.Change saturation\n4.Change hue\n5.Show histogram\n6.Average smoothing\n7.Medain smoothing\n8.Quick median smoothing\n9.Robert outine\n10.Sobel outline\n11.Export and Quit\n"
    )
    if (choice == '0'):
        print("Press any key to close.")
        t = threading.Thread(target=show, args=(filesrc, rgb_img))
        t.setDaemon(True)
        t.start()
    elif (choice == '1'):
        changeContrast(rgb_img)
    elif (choice == '2'):
        changeBrightness(rgb_img)
    elif (choice == '3'):
        changeSaturation(rgb_img)
    elif (choice == '4'):
        changeHue(rgb_img)
    elif (choice == '5'):
        getHistogram(rgb_img)
    elif (choice == '6'):
        avgSmoothing(rgb_img)
    elif (choice == '7'):
        medianSmoothing(rgb_img)
    elif (choice == '8'):
        medianSmoothing_quic(rgb_img)
    elif (choice == '9'):
        robertOutline(rgb_img)
    elif (choice == '10'):
        sobelOutline(rgb_img)
    elif (choice == '11'):
        dest = input("Type in the dest path: ")
        export(dest, rgb_img)
        break
