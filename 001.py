import parser

import cv2
img = cv2.imread('_001.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
value = 30
img_dst = cv2.bilateralFilter(img,value,value*2,value/2)
cv2.imshow("img", img)
cv2.imshow('img_dst',img_dst)
cv2.waitKey(0)
cv2.imwrite('1111.png',img_dst)