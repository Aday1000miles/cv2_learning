import cv2

img = cv2.imread('_001.jpg', cv2.IMREAD_GRAYSCALE)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convoter color

cv2.imshow('gray', img)

cv2.waitKey(0)

#two ways to