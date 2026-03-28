import cv2
capture = cv2.VideoCapture('__001.avi')
if capture.isOpened() == False:
    print('Fail to open the camera!')
    exit(0)

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_fps = capture.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter(
    '__002.avi', fourcc, int(frame_fps), (int(frame_width), int(frame_height)), True)
frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1

while capture.isOpened() and frame_index >= 0:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = capture.read()
    if ret == True:
        cv2.imshow('camera',frame)
        video.write(frame)
        frame_index = frame_index - 1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
video.release()
cv2.destroyAllWindows()


