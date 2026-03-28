import cv2
import matplotlib.pyplot as plt

img = cv2.imread('_006.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

mor = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element)
cv2.imshow('gray', gray)
cv2.imshow('mor', mor)

cv2.waitKey(0)