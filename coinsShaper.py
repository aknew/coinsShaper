#!/usr/local/bin/python3
import numpy as np
import cv2 as cv
import sys
import os
from datetime import datetime

blurSize = 5
thresh = 180
threshType = cv.THRESH_OTSU
minContourSize = 50
debugSave = True

def preprocess(im):
   imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
   imblur = cv.blur(imgray,(blurSize, blurSize))
   ret, imth = cv.threshold(imblur, thresh, 255, threshType)
   return imth

def getContoursRects(th):
    contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rects = list(map(lambda contour: cv.boundingRect(contour), contours))
    rects = filter(lambda rect: rect[2]>minContourSize and rect[3]>minContourSize, rects)
    return list(rects)

def rectCompare(r1, r2):
    s1 = r1[2]*r1[3]
    s2 = r2[2]*r2[3]
    if abs(s1/s2 - 1.0) > 0.2:
        # square ratio is too big, it can't be different side of one object
        return False
    x=max(r1[0], r2[0])
    y=max(r1[1], r2[1])
    w=min(r1[0]+r1[2], r2[0]+r2[2])
    h=min(r1[1]+r1[3], r2[1]+r2[3])

    if x>w or y>h:
        return False

    s=(w-x)*(h-y)

    return s>0

def getCroppedImage(img, rect):
    ofset = blurSize * 3
    s = img.shape
    x1 = max(0, rect[1] - ofset )
    x2 = min(s[0], rect[1]+rect[3] + ofset)
    y1 = max(0, rect[0] - ofset )
    y2 = min(s[1], rect[0]+rect[2] + ofset)
    return img[x1:x2, y1:y2]

t = datetime.now()
path = t.strftime('%Y%m%d %H:%M:%S') + "/"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)

file1 = sys.argv[1]
file2 = sys.argv[2]

im1 = cv.imread(file1)
im2 = cv.imread(file2)

th1= preprocess(im1)
th2= preprocess(im2)

if debugSave:
    cv.imwrite(path + "th1.jpg", th1)
    cv.imwrite(path + "th2.jpg", th2)

r1 = getContoursRects(th1)
r2 = getContoursRects(th2)

number = 0
for rect1 in r1:
    for rect2 in r2:
        if rectCompare(rect1, rect2):
            cropedim1 = getCroppedImage(im1, rect1)
            cropedim2 = getCroppedImage(im2, rect2)

            # from https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv
            h1, w1 = cropedim1.shape[:2]
            h2, w2 = cropedim2.shape[:2]
            #create empty matrix
            vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

            #combine 2 images
            vis[:h1, :w1,:3] = cropedim1
            vis[:h2, w1:w1+w2,:3] = cropedim2
            cv.imwrite(path + "img{}.jpg".format(number), vis)
            number += 1
