import cv2
import numpy as np

#削波
def clipStretch(src, x1, x2):
    src = np.float32(src)
    dst = 255.0 * (src - x1) / (x2 - x1)
    dst[dst > 255] = 255
    dst[dst < 0] = 0
    dst = np.uint8(dst + 0.5)
    return dst

img = cv2.imread(r'Mary.jpg', 0)
cv2.imshow('input', img)

img2 = clipStretch(img, 50, 150)
cv2.imshow('output', img2)
#cv2.imwrite(r'Mary_out.jpg', img2)
cv2.waitKey()
cv2.destroyAllWindows()