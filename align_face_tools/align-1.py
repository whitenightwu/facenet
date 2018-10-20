#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : align-1.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-27:11:08:52
## Description:
## 
##

# Scaling - resize() shirnk or englarge the image# Scali 
import numpy as np
import cv2 as cv
img  = cv.imread('C:\\Users\\amitkumar_kataria\\Desktop\\CV\\Data\\flowers\\messi.jpg',-1)
res = cv.resize(img,None,fx=4, fy=4, interpolation = cv.INTER_CUBIC)
cv.imshow('result',res)
cv.waitKey(0)
cv.destroyAllWindows()
#OR
height, width, _ = img.shape
# when the size of output image is provide fx and fy are calculated from there
res = cv.resize(img,None,fx=1.4, fy=0.8, interpolation = cv.INTER_AREA)
#res = cv.resize(img,None, cv.Size(),0.5,0.5, interpolation = cv.INTER_AREA)
cv.imshow('result',res)
cv.waitKey(0)
cv.destroyAllWindows()

# tRANSLATION cv.wrapAffine() takes 2x3 matrix cv.wrapPerspective takes 3x3 matrix
import numpy as np
import cv2 as cv
img = cv.imread('C:\\Users\\amitkumar_kataria\\Desktop\\CV\\Data\\flowers\\messi.jpg',0)

rows,cols = img.shape

#creating the matrix for inputted to be wrapAffine()
M = np.float32([[1,0,100],[0,1,50]])

# last argumnent for wrapAfine is size of destination image in widhtxheight
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Rotation with 90 degrees and no scaling

img = cv.imread('C:\\Users\\amitkumar_kataria\\Desktop\\CV\\Data\\flowers\\messi.jpg',0)
rows,cols = img.shape

M = cv.getRotationMatrix2D(center= (cols/2,rows/2), angle= 270,scale=2)
#M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dest = cv.warpAffine(src=img,M=M,dsize=(rows*2,cols))
dst = cv.warpAffine(dest,np.float32([[1,0,100],[0,1,10]]),dsize=(rows*2,cols))
cv.imshow('Result',dst)
cv.waitKey(0)
cv.destroyAllWindows()
