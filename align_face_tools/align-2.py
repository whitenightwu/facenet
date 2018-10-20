#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : align-2.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-27:11:09:56
## Description:
## 
##
import cv2
import numpy as np

img = cv2.imread('balckpussy.png', cv2.IMREAD_GRAYSCALE)

rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)

print(M)

M[:,-1] += np.array([(rows - cols)/2 , (cols - rows)/2]).T

#
img2 = cv2.warpAffine(img,M2,(cols,rows))
#

print(M)

print(img.shape)
print(img2.shape)

cv2.imshow('original', img)
cv2.imshow('affine', img2)
cv2.waitKey()

'''
需要注意的是， 在转角度的时候会发生偏移，因此需要用第12行的码进行位置调整
wrapAffine()函数，从左上角为原点旋转，getRotationMatrix2D这个函数有点不靠谱，需要对M[:,-1]这一列需要想一想怎么设置
'''
