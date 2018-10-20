#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : alignemetnt.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-15:10:46:54
## Description:
## 
##
import os
import sys
import cv2
import numpy as np

from PIL import Image
from pylab import *

######################################################
# Img = cv2.imread('/home/ydwu/datasets/tmp2.png')
Img = np.array(Image.open('/home/ydwu/datasets/tmp2.png'))
imshow(Img)
rows, cols, ch = Img.shape

######################################################
# SrcPointsA = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
# CanvasPointsA = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9], [cols * 0.8, rows * 0.9]])

# print 'Please click 3 points'
# SrcPointsA_a = ginput(3)
# print 'you clicked:', SrcPointsA_a
# SrcPointsA_b = []
# for i in range(3):
#     SrcPointsA_b.append(list(SrcPointsA_a[i]))
# print("SrcPointsA_b = ", SrcPointsA_b)
# SrcPointsA = np.float32([SrcPointsA_b])
# print("SrcPointsA = ", SrcPointsA)

SrcPointsA = np.float32([[50.12121212121211, 80.9469696969697], [97.88636363636363, 74.54545454545456], [68.3409090909091, 104.58333333333334]])
# SrcPointsA = np.float32([[52.583333333333314, 77.99242424242425], [99.36363636363637, 76.02272727272728], [54.06060606060605, 126.74242424242426], [91.97727272727272, 126.74242424242426]])
coord5points = [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]

print 'Please click 4 points'
CanvasPointsA_a = ginput(3)
CanvasPointsA_b = []
for i in range(3):
    CanvasPointsA_b.append(list(CanvasPointsA_a[i]))
CanvasPointsA = np.float32([CanvasPointsA_b])
print("CanvasPointsA = ", CanvasPointsA)

######################################################
# PerspectiveMatrix = cv2.getPerspectiveTransform(np.array(SrcPointsA), np.array(CanvasPointsA))
# PerspectiveImg = cv2.warpPerspective(Img, PerspectiveMatrix, (rows, cols))

PerspectiveMatrix = cv2.getAffineTransform(np.array(SrcPointsA), np.array(CanvasPointsA))
PerspectiveImg = cv2.warpAffine(Img, PerspectiveMatrix, (rows, cols))

######################################################
cv2.imshow('align_face', PerspectiveImg)
cv2.waitKey()
