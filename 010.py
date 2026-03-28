import cv2
import matplotlib.pyplot as plt

img = cv2.imread('_001.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



#plt.subplots(121)
#plt.imshow(img[:, :, ::-1])

plt.imshow(img_hsv)
plt.show()
cv2.waitKey(0)