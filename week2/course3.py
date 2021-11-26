import cv2
import numpy as np

def nothing(x):
    pass

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('example2')

cv2.createTrackbar('R ', 'example2', 0, 255, nothing)
cv2.createTrackbar('G ', 'example2', 0, 255, nothing)
cv2.createTrackbar('B ', 'example2', 0, 255, nothing)
cv2.createTrackbar('OFF\ON ', 'example2', 0, 1, nothing)

while(1):
    cv2.imshow('example2', img)
    k = cv2.waitKey(1)
    if(k == ord(' ')):
        break
    b = cv2.getTrackbarPos('B ', 'example2')
    g = cv2.getTrackbarPos('G ', 'example2')
    r = cv2.getTrackbarPos('R ', 'example2')
    s = cv2.getTrackbarPos('OFF\ON ', 'example2')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv2.destroyAllWindows()
