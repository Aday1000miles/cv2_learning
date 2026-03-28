#first(548, 174) -> (0, 0)
#second(330, 403) -> (0, 600)
#third(918, 318) -> (600, 600)
#forth(817, 189) -> (600, 0)

import cv2
import numpy as np

#catch and turn gray
img = cv2.imread('_003.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#resize
img_resize = cv2.resize(gray, None, fx = 0.5, fy = 0.5)

#get weight and hight
(w, h) = img_resize.shape

#build rotation matrix
M = cv2.getRotationMatrix2D((0.5*w, 0.5*h), 45, 1)

#warp(扭曲)
#
img_rotated = cv2.warpAffine(img_resize, M, (w,h))

pts_start = np.float32([[548.0, 174.0], [330.0, 403.0], [918.0, 318.0], [817.0, 189.0]])
pts_end = np.float32([[0.0, 0.0], [0.0, 600.0], [600.0, 600.0], [600.0, 0.0]])

#perspective
M2 = cv2.getPerspectiveTransform(pts_start, pts_end)
img_perspective = cv2.warpPerspective(img_rotated, M2, (600, 600))

#show
cv2.imshow('perspective', img_perspective)
cv2.waitKey(0)
cv2.destroyAllWindows()