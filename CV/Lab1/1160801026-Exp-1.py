import numpy as np
import struct
import math
import sys


def getMin(a, b, c):
    min = a
    if b < min:
        min = b
    if c < min:
        min = c
    return min


class BmpFileHeader:
    def __init__(self):
        self.bfType = 0  # 0x4d42 对应BM
        self.bfSize = 0  # file size
        self.bfReserved1 = 0
        self.bfReserved2 = 0
        self.bfOffBits = 0  # header info offset


class BmpStructHeader:
    def __init__(self):
        self.biSize = 0  # bmpheader size
        self.biWidth = 0
        self.biHeight = 0
        self.biPlanes = 0  # default 1
        self.biBitCount = 0  # one pixel occupy how many img_data
        self.biCompression = 0
        self.biSizeImage = 0
        self.biXPelsPerMeter = 0
        self.biYPelsPerMeter = 0
        self.biClrUsed = 0
        self.biClrImportant = 0


class Bmp(BmpFileHeader, BmpStructHeader):  # extends
    def __init__(self, file_name):
        self.filesrc = file_name
        BmpFileHeader.__init__(self)
        BmpStructHeader.__init__(self)
        self.img_data = []  # pixel array

        file = open(file_name, 'rb')
        # BmpFileHeader
        self.bfType = file.read(2)
        self.bfSize = file.read(4)
        self.bfReserved1 = file.read(2)
        self.bfReserved2 = file.read(2)
        self.bfOffBits = file.read(4)
        # BmpStructHeader
        self.biSize = file.read(4)  # 此结构体的大小
        self.biWidth = file.read(4)
        self.biHeight = file.read(4)
        self.biPlanes = file.read(2)
        self.biBitCount = file.read(2)  # 一像素所占的位数，一般是24
        # pixel size
        self.biCompression = file.read(4)
        self.biSizeImage = file.read(4)
        self.biXPelsPerMeter = file.read(4)
        self.biYPelsPerMeter = file.read(4)
        self.biClrUsed = file.read(4)
        self.biClrImportant = file.read(4)
        #  load pixel info
        countline = 0
        height = int.from_bytes(self.biHeight, 'little')
        width = int.from_bytes(self.biWidth, 'little')
        expect = ((width * int.from_bytes(self.biBitCount, 'little') + 31) &
                  ~31) >> 3
        while countline < height:
            countline += 1
            countcol = 0
            while countcol < width:
                countcol += 1
                bit_count = 0
                while bit_count < (
                        int.from_bytes(self.biBitCount, 'little') // 8):
                    self.img_data.append(file.read(1))
                    bit_count += 1
                if (countcol == width - 1):
                    for i in range(expect - width * 3):
                        file.read(1)  # 跳过填充
        # print(len(self.img_data))
        file.close()

    def generate(self, file_name):
        file = open(file_name, 'wb+')
        # reconstruct File Header
        file.write(self.bfType)
        file.write(self.bfSize)
        file.write(self.bfReserved1)
        file.write(self.bfReserved2)
        file.write(self.bfOffBits)
        # reconstruct bmp header
        file.write(self.biSize)
        file.write(self.biWidth)
        file.write(self.biHeight)
        file.write(self.biPlanes)
        file.write(self.biBitCount)
        file.write(self.biCompression)
        file.write(self.biSizeImage)
        file.write(self.biXPelsPerMeter)
        file.write(self.biYPelsPerMeter)
        file.write(self.biClrUsed)
        file.write(self.biClrImportant)
        # reconstruct img_data
        width = int.from_bytes(self.biWidth, 'little')
        expect = ((width * int.from_bytes(self.biBitCount, 'little') + 31) &
                  ~31) >> 3
        # print(expect)
        count = 0
        for bit in self.img_data:
            count += 1
            if (count != width * 3):
                file.write(bit)
            else:
                file.write(bit)
                count = 0
                for i in range(expect - width * 3):
                    tmp = struct.pack('B', 0)
                    file.write(tmp)
        file.close()

    def convert(self):
        B = []
        G = []
        R = []
        count = 0
        origin_img_data = self.img_data.copy()  # origin_img_data

        # YIQ
        print('Converting to YIQ')
        for img_byte in origin_img_data:
            count += 1
            if count % 3 == 1:
                B.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 2:
                G.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 0:
                R.append(int.from_bytes(img_byte, 'little'))

        Y = (0.299 * np.mat(R) + 0.587 * np.mat(G) +
             0.114 * np.mat(B)).tolist()[0]
        I = (0.596 * np.mat(R) - 0.274 * np.mat(G) - 0.322 * np.mat(B) +
             128).tolist()[0]
        Q = (0.211 * np.mat(R) - 0.523 * np.mat(G) + 0.312 * np.mat(B) +
             128).tolist()[0]
        for i in range(int(len(self.img_data) / 3)):
            self.img_data[3 * i] = struct.pack('B', int(Y[i]) & 0xFF)
            self.img_data[3 * i + 1] = struct.pack('B', int(I[i]) & 0xFF)
            self.img_data[3 * i + 2] = struct.pack('B', int(Q[i]) & 0xFF)
        dest = self.filesrc.split('.bmp')[0] + '-1160801026-YIQ.bmp'
        self.generate(dest)

        # YCbCr
        print('Converting to YCbCr')
        for img_byte in origin_img_data:
            count += 1
            if count % 3 == 1:
                B.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 2:
                G.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 0:
                R.append(int.from_bytes(img_byte, 'little'))

        Y = (0.299 * np.mat(R) + 0.587 * np.mat(G) +
             0.114 * np.mat(B)).tolist()[0]
        Cr = (0.500 * np.mat(R) - 0.419 * np.mat(G) - 0.081 * np.mat(B) +
              128).tolist()[0]
        Cb = (-0.169 * np.mat(R) - 0.331 * np.mat(G) + 0.500 * np.mat(B) +
              128).tolist()[0]
        for i in range(int(len(self.img_data) / 3)):
            self.img_data[3 * i] = struct.pack('B', int(Y[i]) & 0xFF)
            self.img_data[3 * i + 1] = struct.pack('B', int(Cb[i]) & 0xFF)
            self.img_data[3 * i + 2] = struct.pack('B', int(Cr[i]) & 0xFF)
        dest = self.filesrc.split('.bmp')[0] + '-1160801026-YCbCr.bmp'
        self.generate(dest)

        # XYZ
        print('Converting to XYZ')
        for img_byte in origin_img_data:
            count += 1
            if count % 3 == 1:
                B.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 2:
                G.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 0:
                R.append(int.from_bytes(img_byte, 'little'))

        X = (0.412453 * np.mat(R) + 0.357580 * np.mat(G) +
             0.180423 * np.mat(B)).tolist()[0]
        Y = (0.212671 * np.mat(R) + 0.715160 * np.mat(G) +
             0.072169 * np.mat(B)).tolist()[0]
        Z = (0.019334 * np.mat(R) + 0.119193 * np.mat(G) +
             0.990 * np.mat(B)).tolist()[0]
        for i in range(int(len(self.img_data) / 3)):
            self.img_data[3 * i] = struct.pack('B', int(X[i]) & 0xFF)
            self.img_data[3 * i + 1] = struct.pack('B', int(Y[i]) & 0xFF)
            self.img_data[3 * i + 2] = struct.pack('B', int(Z[i]) & 0xFF)
        dest = self.filesrc.split('.bmp')[0] + '-1160801026-XYZ.bmp'
        self.generate(dest)

        # HSI
        print('Converting to HSI')
        for img_byte in origin_img_data:
            count += 1
            if count % 3 == 1:
                B.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 2:
                G.append(int.from_bytes(img_byte, 'little'))
            elif count % 3 == 0:
                R.append(int.from_bytes(img_byte, 'little'))

        I = ((np.mat(R) + np.mat(G) + np.mat(B)) / 3).tolist()[0]
        S = []
        H = []
        for i in range(int(len(self.img_data) / 3)):
            if R[i] + G[i] + B[i] != 0:
                # 如想使用归一到0~255的公式，则将100改成255
                S.append(100 * (1 - (3 * getMin(R[i], G[i], B[i])) /
                                (R[i] + G[i] + B[i])))
            else:
                S.append(100)  # 如想使用归一到0~255的公式，则将100改成255
            den = np.sqrt((R[i] - G[i])**2 + (R[i] - B[i]) * (G[i] - B[i]))
            if den != 0 and not math.isnan(den):
                thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
                if G[i] < B[i]:
                    thetha = 2 * math.pi - thetha
                H.append(
                    (thetha / 2 * math.pi) * 360)  # 如想使用归一到0~255的公式，则将360改成255
            else:
                H.append(0)

        for i in range(int(len(self.img_data) / 3)):
            self.img_data[3 * i] = struct.pack('B', int(H[i]) & 0xFF)
            self.img_data[3 * i + 1] = struct.pack('B', int(S[i]) & 0xFF)
            self.img_data[3 * i + 2] = struct.pack('B', int(I[i]) & 0xFF)
        dest = self.filesrc.split('.bmp')[0] + '-1160801026-HSI.bmp'
        self.generate(dest)


src = sys.argv[1]
bmp_img = Bmp(src)
bmp_img.convert()
