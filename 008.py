import cv2
import matplotlib.pyplot as plt

img = cv2.imread('_005.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

med = cv2.medianBlur(gray, 3)

#图片阈值分割，白黑，背景为白，前景为黑
ret, bin1 = cv2.threshold(med, 150, 255, cv2.THRESH_BINARY_INV)
#二值化BINARY
#反色INV
cv2.imshow('bin1',bin1)

ret, bin2 = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)
ret, bin3 = cv2.threshold(bin2, 150, 255, cv2.THRESH_BINARY_INV)
#二值化BINARY
#大金算法OTSU
cv2.imshow('bin',bin3)

bin4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 31)
bin5 = cv2.medianBlur(bin4, 3)
cv2.imshow('bin',bin5)


cv2.waitKey(0)