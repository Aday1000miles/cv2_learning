import time

import cv2
capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    print('Fail to open the camera!')
    exit(0)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_fps = capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('__001.avi', fourcc, int(frame_fps), (int(frame_width), int(frame_height)), True)
while capture.isOpened():
    ret, frame = capture.read()
    if ret == True:
        cv2.imshow('camera',frame)
        video.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()


