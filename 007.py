import cv2
import matplotlib.pyplot as plt

img = cv2.imread('_004.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#均值滤波
blur = cv2.blur(gray,(9,9))

#高斯均值滤波
gau = cv2.GaussianBlur(gray, (9,9), 0)

#中值滤波,专门解决椒盐噪声
med = cv2.medianBlur(gray, 3)

#双边滤波
#pass



cv2.imshow('img', img)
cv2.imshow('blur', blur)
cv2.imshow('gau', gau)
cv2.imshow('med', med)




#图片阈值分割，白黑，背景为白，前景为黑
ret, bin1 = cv2.threshold(med, 150, 255, cv2.THRESH_BINARY_INV)
#二值化BINARY
#反色INV
cv2.imshow('bin1',bin1)

ret, bin2 = cv2.threshold(med, 150, 255, cv2.THRESH_OTSU)
#二值化BINARY
#
cv2.imshow('bin',bin2)


can = cv2.Canny(med, 40, 120)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

mor = cv2.morphologyEx(can, cv2.MORPH_ERODE, element)

cv2.imshow('mor', mor)

cv2.waitKey(0)