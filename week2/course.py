import numpy as np
import cv2

# <editor-fold desc="读取保存图片">
img = cv2.imdecode(np.fromfile(r'test02img.jpg'\
    , dtype=np.uint8), cv2.IMREAD_UNCHANGED)

cv2.namedWindow('example', 0)
cv2.imshow('example', img)

cv2.waitKey(0)

cv2.imwrite('saved2_test02img.jpg',img)
cv2.imencode('.jpg', img)[1].tofile('saved_test02img.jpg')
# </editor-fold>