import cv2

img = cv2.imread('_001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resize = cv2.resize(gray, None, fx = 0.5, fy = 0.5)
#cv2.imshow('resize', img_resize)

#get weight and hight of the img
(w, h) = gray.shape

#get center of the img
#(center point, angle, resize of linear)
M = cv2.getRotationMatrix2D((0.5*w, 0.5*h), 45, 1)

#constant
img_rotated1 = cv2.warpAffine(img_resize, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=1000)
cv2.imshow('rotated1', img_rotated1)
cv2.waitKey(0)

#reflect
img_rotated2 = cv2.warpAffine(img_resize, M, (w,h), borderMode=cv2.BORDER_REFLECT, borderValue=1000)
cv2.imshow('rotated2', img_rotated2)
cv2.waitKey(0)

#copy
img_rotated3 = cv2.warpAffine(img_resize, M, (w,h), borderMode=cv2.BORDER_REPLICATE, borderValue=1000)
cv2.imshow('rotated3', img_rotated3)
cv2.waitKey(0)

#turn 180 degree
M2 = cv2.getRotationMatrix2D((0.5*w, 0.5*h), 180, 1)
img_rotated4 = cv2.warpAffine(img, M2, (w,h))
cv2.imshow('rotated4', img_rotated4)
cv2.waitKey(0)