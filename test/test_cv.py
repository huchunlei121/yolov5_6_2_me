import math
import random

import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    path = "../data/images/bus.jpg"
    img = cv2.imread(path)
    h, w, _ = img.shape
    width, height = w, h
    shear = 8

    C = np.eye(3)
    C[0, 2] = -w / 4  # x translation (pixels)
    C[1, 2] = -h / 4  # y translation (pixels)

    R = np.eye(3)
    a = random.uniform(-10, 10)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - 0.7, 1 + 0.1)
    # s = 2 ** random.uniform(-scale, scale)
    # 图片旋转得到仿射变化矩阵赋给R的前两行
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    M = C

    im1 = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
    im2 = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    Image.fromarray(im1[:, :, ::-1]).show()
    Image.fromarray(im2[:, :, ::-1]).show()